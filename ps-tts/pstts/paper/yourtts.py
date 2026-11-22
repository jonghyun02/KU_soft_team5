"""Your-TTS 프런트엔드: 학습한 Coqui VITS(Your-TTS) 체크포인트에서 논문 값을 꺼낸다.

식 (3), (4) 의 duration_TTS, 3.2절의 토큰별 길이 d_i, 식 (8) 의 텍스트 인코더 평균 m_p 를 coqui-tts 0.28.0 의 Vits 에서
그대로 얻는다. 길이 경로는 설치된 TTS/tts/models/vits.py 의 Vits.inference 와 같다.

    g = Vits._get_speaker_conditioning(aux_input, "emb_g", normalize_embedding=False)
        # TTS/tts/models/base_tts.py BaseTTS._get_speaker_conditioning: d-vector 는 F.normalize(L2) 뒤 [B, C, 1]
    lang_emb = Vits.emb_l(language_ids).unsqueeze(-1)          # use_language_embedding 일 때
    x, m_p, logs_p, x_mask = Vits.text_encoder(x, x_lengths, lang_emb=lang_emb)
        # TTS/tts/layers/vits/networks.py TextEncoder.forward: stats = proj(x) * x_mask 를 m_p, logs_p 로 나눔
    logw = Vits.duration_predictor(x, x_mask, g=g if condition_dp_on_speaker else None, reverse=True,
                                   noise_scale=inference_noise_scale_dp, lang_emb=lang_emb)
        # use_sdp 면 StochasticDurationPredictor(reverse=True). 아니면 DurationPredictor(reverse, 잡음 없음)
    w = exp(logw) * x_mask * length_scale

Vits.inference 는 이 w 를 올림(ceil)해 정렬 경로를 만든다. 여기서는 올리기 전 실수 w 를 쓴다(명세 3.4. 논문은 정수 길이의
한계를 들어 미분 가능한 길이 예측기 [39] 를 썼다). 모델의 inference_noise_scale_dp 와 length_scale 은 PaperConfig 의
noise_scale_dp(기본 0, 해석: 확률 길이 예측기를 결정적으로)와 length_scale(기본 1)로 맞춰 둔다. 초 = w x hop / sample_rate.
문장은 하나씩(batch 1) 부른다. Vits.inference 가 x_lengths 없이 부를 때와 같은 꼴이다.

토큰은 g2p.tokens 꼴(빈 토큰 g2p.BLANK 포함)이다. BLANK 는 모델의 blank id 로, 나머지 글자는 모델 글자 목록의 id 로
바꾼다(Coqui TTSTokenizer.encode 다음 intersperse_blank_char 와 같은 열). 목록에 없는 글자가 있으면 그 글자들을 적어
UnknownTokenError 를 낸다(Coqui 는 버리고 넘어가지만 그러면 토큰별 길이가 토큰 열과 어긋난다). 미리 거르려면
unknown_tokens() 를 쓴다.

모델을 올리기 전에 두 가지를 확인한다. 틀리면 ISO, PS 에서 LaBSE 와 모델 2 를 부르기 전에 멈춘다.
- G2P: 체크포인트 폴더의 g2p.json(학습 때 phonemize_manifest.py 가 쓴 G2P 설정)이 PaperConfig 의 g2p_en_voice,
  g2p_en_stress 와 같아야 한다. 다르면 토큰 열이 학습 때와 달라진다.
- 모음표(식 8, 9): scripts/paper_build_vowel_table.py 가 이 프런트엔드의 encoder_means 로 만든 vowel_table.json 을 읽고,
  표를 만든 체크포인트(sha1)가 지금 체크포인트와 다르면 멈춘다. 표를 만드는 쪽은 load_table=False 로 올린다.

화자 d-vector: 발화별 d-vector 파일은 LibriTTS-360 만으로도 디스크 약 545 MB, 올리면 약 2.9 GB, 30~50초가 든다(Coqui
SpeakerManager 가 벡터를 파이썬 실수 목록으로 들고 있음). 추론에는 화자 하나의 평균만 쓰므로 train_yourtts.py 가 화자마다
평균 하나를 담은 d_vector_means.pth(SPEAKER_MEANS_NAME)를 학습 폴더에 쓰고, 이 프런트엔드는 그 파일이 있으면 그것을 올린다.
값은 발화별 파일로 Coqui get_mean_embedding 을 부른 것과 같다. --tts-speaker 로 발화(클립) 키를 주면 발화별 파일을 쓴다.
"""
from __future__ import annotations

import gc
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from ..scorers import resolve_device
from . import g2p
from .config import PaperConfig
from .vowel_embed import EmbedVowelTable, file_sha1, load_vowel_table

_BEST_STEP = re.compile(r"^best_model_(\d+)\.pth$")
VOWEL_TABLE_NAME = "vowel_table.json"
SPEAKER_MEANS_NAME = "d_vector_means.pth"


class UnknownTokenError(ValueError):
    """모델 글자 목록에 없는 토큰이 있을 때."""

    def __init__(self, tokens: Sequence[str], lang: str) -> None:
        self.tokens = list(tokens)
        self.lang = lang
        shown = ", ".join(repr(t) for t in self.tokens[:20])
        more = f" 외 {len(self.tokens) - 20}개" if len(self.tokens) > 20 else ""
        super().__init__(f"모델 글자 목록에 없는 토큰이 있습니다({lang}): {shown}{more}")


# ---------------------------------------------------------------------------- 체크포인트 폴더
def find_checkpoint(tts_dir: Path, name: str | None = None) -> Path:
    """체크포인트 파일. name 이 없으면 best_model.pth, 가장 큰 걸음의 best_model_N.pth, model_file.pth 순서로 찾는다."""
    if name:
        p = Path(name)
        if not p.is_absolute():
            p = tts_dir / p
        if not p.is_file():
            raise FileNotFoundError(f"체크포인트가 없습니다: {p}")
        return p
    best = tts_dir / "best_model.pth"
    if best.is_file():
        return best
    steps = [(int(m.group(1)), p) for p in tts_dir.glob("best_model_*.pth") if (m := _BEST_STEP.match(p.name))]
    if steps:
        return max(steps)[1]
    released = tts_dir / "model_file.pth"
    if released.is_file():
        return released
    others = sorted(p.name for p in tts_dir.glob("*.pth"))
    raise FileNotFoundError(f"{tts_dir} 에 best_model*.pth 나 model_file.pth 가 없습니다(있는 .pth: {others}). "
                            "다른 파일은 --tts-checkpoint 로 고르세요.")


def _resolve(value: str, tts_dir: Path) -> Path | None:
    """경로가 그대로 있으면 그것, 아니면 tts_dir 기준 상대 경로, 아니면 tts_dir 안 같은 이름 파일."""
    p = Path(value)
    for cand in ([p] if p.is_absolute() else []) + [tts_dir / p, tts_dir / p.name]:
        if cand.is_file():
            return cand.resolve()
    return None


def fix_config_paths(config: Any, tts_dir: Path) -> dict[str, list]:
    """config 의 파일 경로 칸을 tts_dir 기준으로 고친다. 고친 칸 {이름: [원래, 새]} 를 돌려준다.

    d_vector_file 과 language_ids_file 은 쓰는 경우 꼭 있어야 한다. 학습 때만 쓰는 화자 인코더 경로(화자 일관성 손실)는
    길이와 m_p 에 쓰지 않으므로 비운다.
    """
    changes: dict[str, list] = {}
    use_lang = bool(config.model_args.get("use_language_embedding"))
    for owner_name, owner in (("model_args", config.model_args), ("config", config)):
        for key in ("d_vector_file", "language_ids_file", "speakers_file"):
            if not hasattr(owner, key):
                continue
            val = getattr(owner, key)
            if not val:
                continue
            values = list(val) if isinstance(val, (list, tuple)) else [val]
            fixed: list[str] = []
            for v in values:
                r = _resolve(str(v), tts_dir)
                if r is None:
                    needed = (key == "d_vector_file" and bool(config.model_args.get("use_d_vector_file"))) or \
                             (key == "language_ids_file" and use_lang)
                    if needed:
                        raise FileNotFoundError(f"config 의 {owner_name}.{key} 파일을 찾지 못했습니다: {v} "
                                                f"({tts_dir} 안에 같은 이름 파일도 없음)")
                    continue
                fixed.append(str(r))
            new: Any = (fixed if isinstance(val, (list, tuple)) else (fixed[0] if fixed else None))
            if new != val:
                changes[f"{owner_name}.{key}"] = [val, new]
                setattr(owner, key, new)
    args = config.model_args
    if args.get("speaker_encoder_model_path") or args.get("use_speaker_encoder_as_loss"):
        changes["model_args.speaker_encoder"] = [args.get("speaker_encoder_model_path"), ""]
        args.speaker_encoder_model_path = ""
        args.speaker_encoder_config_path = ""
        args.use_speaker_encoder_as_loss = False
    return changes


def write_speaker_means(d_vector_file: str | Path, out: str | Path) -> dict[str, Any]:
    """발화별 d-vector 파일에서 화자마다 평균 벡터 하나를 담은 작은 추론용 파일(Coqui d_vector_file 꼴)을 쓴다.

    키와 name 은 화자 이름이고 n_clips 에 발화 수를 적는다. 평균은 Coqui SpeakerManager.get_mean_embedding 과 같게
    파일 순서대로 쌓은 float64 배열의 평균이다.
    """
    from TTS.tts.utils.managers import load_file, save_file

    data = load_file(str(d_vector_file))
    by_name: dict[str, list[Any]] = {}
    for entry in data.values():
        by_name.setdefault(str(entry["name"]), []).append(entry["embedding"])
    if not by_name:
        raise ValueError(f"d-vector 파일에 벡터가 없습니다: {d_vector_file}")
    means = {name: {"name": name, "embedding": np.stack(vecs).mean(0).tolist(), "n_clips": len(vecs)}
             for name, vecs in sorted(by_name.items())}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    save_file(means, str(tmp))
    tmp.replace(target)
    return {"path": str(target.resolve()), "speakers": len(means), "clips": len(data)}


def use_speaker_means(config: Any, tts_dir: Path, wanted: str | None) -> list[str] | None:
    """tts_dir 에 화자 평균 파일이 있고 고른 화자가 거기 있으면 config 의 d_vector_file 을 그 파일로 바꾼다.

    바꿨으면 [원래 경로, 새 경로], 아니면 None. 발화(클립) 키를 고르면 발화별 파일을 그대로 쓴다.
    """
    means = tts_dir / SPEAKER_MEANS_NAME
    if not (config.model_args.get("use_d_vector_file") and means.is_file()):
        return None
    if wanted is not None:
        from TTS.tts.utils.managers import load_file

        if wanted not in load_file(str(means)):
            return None
    before = config.model_args.get("d_vector_file")
    new = str(means.resolve())
    config.model_args.d_vector_file = new
    if hasattr(config, "d_vector_file"):
        config.d_vector_file = new
    return [str(before), new]


def check_g2p_file(tts_dir: Path, en_voice: str, en_stress: bool) -> dict[str, Any]:
    """체크포인트 폴더의 g2p.json(학습 때 G2P)이 지금 설정과 같은지 본다. 없거나 다르면 한국어 오류."""
    path = tts_dir / g2p.SETTINGS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} 가 없습니다. 학습 매니페스트 폴더에서 phonemize_manifest.py 가 쓴 {g2p.SETTINGS_NAME} 를 "
                                "체크포인트 폴더에 두세요(train_yourtts.py 는 학습 폴더에 복사함).")
    doc = g2p.read_settings(path)
    g2p.check_settings(doc, en_voice, en_stress, where=str(path))
    return doc


def vowel_table_path(cfg: PaperConfig, tts_dir: Path) -> Path:
    """모음표 JSON 경로: --vowel-table 이 있으면 그것, 없으면 tts_dir/vowel_table.json."""
    return Path(cfg.vowel_table) if cfg.vowel_table else tts_dir / VOWEL_TABLE_NAME


def read_vowel_table(path: Path, checkpoint_sha1: str) -> EmbedVowelTable:
    """모음표를 읽고 같은 체크포인트(sha1)로 만든 표인지 확인한다. 없거나 다르면 한국어 오류."""
    if not path.is_file():
        raise FileNotFoundError(f"모음표(식 8, 9)가 없습니다: {path}. "
                                "scripts\\paper_build_vowel_table.py 로 먼저 만드세요.")
    table = load_vowel_table(path)
    made_by = (table.meta.get("checkpoint") or {}).get("sha1")
    if made_by != checkpoint_sha1:
        raise ValueError(f"모음표 {path} 는 다른 체크포인트(sha1 {made_by})로 만든 것입니다. "
                         f"지금 체크포인트 sha1 {checkpoint_sha1} 로 다시 만드세요.")
    return table


class CheckedFolder(NamedTuple):
    """check_folder 가 확인한 체크포인트 폴더."""
    tts_dir: Path
    config_path: Path
    checkpoint: Path
    checkpoint_sha1: str
    en_voice: str
    en_stress: bool
    g2p_settings: dict[str, Any]
    table: EmbedVowelTable | None       # load_table=False 면 None


def check_folder(cfg: PaperConfig, *, load_table: bool = True) -> CheckedFolder:
    """모델을 올리기 전에 체크포인트 폴더의 가벼운 준비물을 확인한다: config.json, 체크포인트, g2p.json(학습 때 G2P 가
    지금 설정과 같은지), 모음표(같은 체크포인트로 만든 것). 파이프라인은 모델 1 을 올리기 전에도 부른다(pipeline.preflight)."""
    if not cfg.tts_dir:
        raise ValueError("Your-TTS 프런트엔드에는 체크포인트 폴더(--tts-dir)가 필요합니다.")
    tts_dir = Path(cfg.tts_dir).resolve()
    config_path = tts_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json 이 없습니다: {config_path}")
    ckpt = find_checkpoint(tts_dir, cfg.tts_checkpoint)
    en_voice = g2p.check_en_voice(cfg.g2p_en_voice)
    en_stress = bool(cfg.g2p_en_stress)
    settings = check_g2p_file(tts_dir, en_voice, en_stress)
    sha1 = file_sha1(ckpt)
    table = read_vowel_table(vowel_table_path(cfg, tts_dir), sha1) if load_table else None
    return CheckedFolder(tts_dir, config_path, ckpt, sha1, en_voice, en_stress, settings, table)


def _check_config(config: Any) -> None:
    """논문 판 레시피(train_yourtts.py)로 학습한 모델인지 본다: 미리 음소화한 글, 빈 토큰, 정리기 없음."""
    problems = []
    if str(config.get("model", "")).lower() != "vits":
        problems.append(f"model={config.get('model')} (vits 여야 함)")
    if not config.get("add_blank"):
        problems.append("add_blank=False (논문의 빈 토큰 b 가 필요함)")
    if config.get("use_phonemes"):
        problems.append("use_phonemes=True (논문 판은 g2p 로 미리 음소화한 글로 학습함)")
    if config.get("text_cleaner"):
        problems.append(f"text_cleaner={config.get('text_cleaner')} (정리기 없이 학습해야 토큰이 같음)")
    if config.get("enable_eos_bos_chars"):
        problems.append("enable_eos_bos_chars=True (시작·끝 토큰이 없어야 함)")
    if config.model_args.get("use_speaker_embedding") and not config.model_args.get("use_d_vector_file"):
        problems.append("화자 임베딩 층 모델(이 프런트엔드는 d-vector 모델과 화자 없는 모델만 씀)")
    if problems:
        raise ValueError("이 체크포인트는 논문 판 Your-TTS 레시피와 맞지 않습니다: " + "; ".join(problems))


# ---------------------------------------------------------------------------- 프런트엔드
class YourTTSFrontend:
    """학습한 Your-TTS 체크포인트(tts_dir 의 config.json + .pth)로 길이, m_p, 모음표 비용을 준다."""

    name = "yourtts"

    def __init__(self, cfg: PaperConfig, device: str, *, load_table: bool = True) -> None:
        """load_table=False 는 모음표를 만드는 쪽(paper_build_vowel_table.py)만 쓴다. 그때는 labels, cost 를 처음 부를 때 읽는다."""
        if not cfg.tts_dir:
            raise ValueError("Your-TTS 프런트엔드에는 체크포인트 폴더(--tts-dir)가 필요합니다.")
        self.cfg = cfg
        self.device = resolve_device(device, cfg.cpu_threads)     # cuda 를 못 쓰면 멈춘다(조용한 CPU 전환 없음)
        # 모델을 올리기 전에 G2P 와 모음표를 확인한다(틀리면 LaBSE, 모델 2 를 부르기 전에 멈춤)
        folder = check_folder(cfg, load_table=load_table)
        self.tts_dir = folder.tts_dir
        config_path = folder.config_path
        ckpt = folder.checkpoint
        self.en_voice, self.en_stress, self.g2p_settings = folder.en_voice, folder.en_stress, folder.g2p_settings
        self.checkpoint_path = str(ckpt)
        self.checkpoint_sha1 = folder.checkpoint_sha1
        self._table: EmbedVowelTable | None = folder.table

        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits

        config = VitsConfig()
        config.load_json(str(config_path))
        self.path_changes = fix_config_paths(config, self.tts_dir)
        _check_config(config)
        means = use_speaker_means(config, self.tts_dir, cfg.tts_speaker)
        if means is not None:
            self.path_changes["model_args.d_vector_file(화자 평균)"] = means
        self._speaker_means = means is not None
        model = Vits(config)
        model.load_checkpoint(config, str(ckpt), eval=True)
        # Vits.inference 가 쓰는 두 값을 PaperConfig 로 맞춘다(같은 모델로 inference 를 불러도 같은 w 가 나오게)
        model.inference_noise_scale_dp = float(cfg.noise_scale_dp)
        model.length_scale = float(cfg.length_scale)
        self.model = model.to(self.device)
        self.config = config
        self.config_path = config_path
        self.sample_rate = int(config.audio.sample_rate)
        self.hop_length = int(config.audio.hop_length)
        self.frame_sec = self.hop_length / self.sample_rate

        tok = model.tokenizer
        if tok.blank_id is None:
            raise ValueError("모델 글자 목록에 빈 토큰(blank)이 없습니다.")
        self.blank_id = int(tok.blank_id)
        self._char_to_id: dict[str, int] = {c: i for i, c in enumerate(tok.characters.vocab)
                                            if i not in (self.blank_id, tok.pad_id)}
        self._speaker = self._pick_speaker(cfg.tts_speaker)
        self._lang_ids = self._language_ids(cfg)
        self.missing_vowels: dict[str, Counter[str]] = {"en": Counter(), "ko": Counter()}

    # ------------------------------------------------------------------ 화자, 언어
    def _pick_speaker(self, wanted: str | None) -> dict[str, Any]:
        """화자 이름이나 d-vector 키(클립). 없으면 이름순 첫 화자(그 화자 d-vector 평균, Coqui 와 같은 방식).

        화자 평균 파일(d_vector_means.pth)을 올렸으면 화자마다 벡터가 이미 평균 하나다. 발화 수는 그 파일의 n_clips.
        """
        model = self.model
        if not model.args.use_d_vector_file:
            return {"kind": "none", "tensor": None}
        sm = model.speaker_manager
        if sm is None or not sm.embeddings:
            raise ValueError("d-vector 파일에 벡터가 없습니다.")
        names = sorted(sm.embedding_names)
        if wanted is None or wanted in sm.embeddings_by_names:
            name = names[0] if wanted is None else wanted
            vec, kind, key = sm.get_mean_embedding(name, num_samples=None, randomize=False), "speaker_mean", None
        elif wanted in sm.embeddings:
            name, kind, key = sm.embeddings[wanted]["name"], "clip", wanted
            vec = sm.embeddings[wanted]["embedding"]
        else:
            raise ValueError(f"화자를 찾지 못했습니다: {wanted} (화자 이름 예: {names[:5]})")
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != int(model.args.d_vector_dim):
            raise ValueError(f"d-vector 차원 {arr.shape[1]} 이 모델 d_vector_dim {model.args.d_vector_dim} 과 다릅니다.")
        if self._speaker_means:
            n = int(sm.embeddings[name].get("n_clips", 1))
        else:
            n = len(sm.embeddings_by_names.get(name, []))
        return {"kind": kind, "name": name, "key": key, "n_clips": n,
                "d_vector_file": "speaker_means" if self._speaker_means else "per_clip",
                "tensor": torch.from_numpy(arr).to(self.device)}

    def _language_ids(self, cfg: PaperConfig) -> dict[str, torch.Tensor | None]:
        model = self.model
        if not (model.args.use_language_embedding and hasattr(model, "emb_l")):
            return {"en": None, "ko": None}
        table = model.language_manager.name_to_id
        out: dict[str, torch.Tensor | None] = {}
        for code, name in (("en", cfg.tts_lang_en), ("ko", cfg.tts_lang_ko)):
            if name not in table:
                raise ValueError(f"모델 언어 목록에 {name} 이 없습니다({code}). 있는 언어: {sorted(table)}")
            out[code] = torch.tensor([int(table[name])], dtype=torch.long, device=self.device)
        return out

    # ------------------------------------------------------------------ 토큰
    def tokens(self, text: str, lang: str) -> list[str]:
        """g2p.tokens(text, lang): 빈 토큰을 끼운 토큰 열(S_b, T_b). 영어 목소리와 강세표는 cfg(= g2p.json)대로."""
        return g2p.tokens(text, lang, en_voice=self.en_voice, en_stress=self.en_stress)

    def unknown_tokens(self, toks: Sequence[str]) -> list[str]:
        """모델 글자 목록에 없는 토큰(겹침 없이, 나온 순서). BLANK 는 빼고 본다."""
        seen: dict[str, None] = {}
        for t in toks:
            if t != g2p.BLANK and t not in self._char_to_id:
                seen.setdefault(t, None)
        return list(seen)

    def ids(self, toks: Sequence[str], lang: str) -> list[int]:
        """토큰 -> 모델 id. BLANK 는 blank id. 목록에 없는 글자가 있으면 UnknownTokenError."""
        bad = self.unknown_tokens(toks)
        if bad:
            raise UnknownTokenError(bad, g2p.check_lang(lang))
        return [self.blank_id if t == g2p.BLANK else self._char_to_id[t] for t in toks]

    def vocab_vowels(self, lang: str) -> list[str]:
        """모델 글자 목록에 있는 모음 토큰(한국어 중성 자모 또는 영어 IPA 모음 글자)."""
        code = g2p.check_lang(lang)
        return sorted(c for c in self._char_to_id if len(c) == 1 and g2p.is_vowel(c, code))

    # ------------------------------------------------------------------ 모델 부르기 (Vits.inference 와 같은 경로)
    @torch.inference_mode()
    def _encode(self, toks: Sequence[str], lang: str) -> tuple:
        code = g2p.check_lang(lang)
        if len(toks) == 0:
            raise ValueError("토큰 열이 비었습니다(빈 글도 g2p.tokens 는 [BLANK] 를 줌).")
        model = self.model
        x = torch.tensor([self.ids(toks, code)], dtype=torch.long, device=self.device)
        x_lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
        aux = {"d_vectors": self._speaker["tensor"], "speaker_ids": None, "language_ids": self._lang_ids[code]}
        g = model._get_speaker_conditioning(aux, "emb_g", normalize_embedding=False)
        lang_emb = None
        lid = aux["language_ids"]
        if model.args.use_language_embedding and lid is not None:
            lang_emb = model.emb_l(lid).unsqueeze(-1)
        h, m_p, _logs_p, x_mask = model.text_encoder(x, x_lengths, lang_emb=lang_emb)
        return h, m_p, x_mask, g, lang_emb

    @torch.inference_mode()
    def _frames(self, toks: Sequence[str], lang: str) -> np.ndarray:
        model = self.model
        h, _m_p, x_mask, g, lang_emb = self._encode(toks, lang)
        g_dp = g if model.args.condition_dp_on_speaker else None
        if model.args.use_sdp:
            logw = model.duration_predictor(h, x_mask, g=g_dp, reverse=True,
                                            noise_scale=model.inference_noise_scale_dp, lang_emb=lang_emb)
        else:
            logw = model.duration_predictor(h, x_mask, g=g_dp, lang_emb=lang_emb)
        w = torch.exp(logw) * x_mask * model.length_scale
        return w[0, 0].double().cpu().numpy()

    def duration_frames(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        """열마다 토큰별 길이(프레임, 실수 w). Vits.inference 가 올림하기 전 값."""
        return [self._frames(seq, lang) for seq in seqs]

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        """열마다 토큰별 길이(초) = w x hop / sample_rate. 빈 토큰도 길이를 받는다(논문과 같음)."""
        return [f * self.frame_sec for f in self.duration_frames(seqs, lang)]

    def encoder_means(self, toks: list[str], lang: str) -> np.ndarray:
        """텍스트 인코더 출력에 선형 투영(proj)을 씌워 나눈 평균 m_p. 꼴 (C, 토큰 수)."""
        _h, m_p, _x_mask, _g, _lang_emb = self._encode(toks, lang)
        return m_p[0].double().cpu().numpy()

    # ------------------------------------------------------------------ 모음표
    @property
    def vowel_table_path(self) -> Path:
        return vowel_table_path(self.cfg, self.tts_dir)

    @property
    def table(self) -> EmbedVowelTable:
        """식 (8), (9) 모음표. 보통은 만들 때 읽어 두고, load_table=False 로 만들었으면 처음 쓸 때 읽는다."""
        if self._table is None:
            self._table = read_vowel_table(self.vowel_table_path, self.checkpoint_sha1)
        return self._table

    def labels(self, toks: list[str], lang: str) -> list[str | None]:
        """표에 있는 모음 토큰은 그 토큰 자체, 아니면 None(null). 표에 없는 모음은 null 로 보고 missing_vowels 에 센다."""
        code = g2p.check_lang(lang)
        allowed = set(self.table.symbols(code))
        out: list[str | None] = []
        for t in toks:
            if g2p.is_vowel(t, code):
                if t in allowed:
                    out.append(t)
                    continue
                self.missing_vowels[code][t] += 1
            out.append(None)
        return out

    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray:
        """(len(src), len(tgt)) 국소 비용. 모음끼리 식 (9) 거리, null 끼리 0, null 과 모음 사이 표의 최대 거리(해석)."""
        return self.table.cost(src_labels, tgt_labels)

    # ------------------------------------------------------------------ 기록
    def info(self) -> dict:
        import importlib.metadata as md

        cfg, model = self.cfg, self.model
        spk = {k: v for k, v in self._speaker.items() if k != "tensor"}
        table_info: dict[str, Any] | None = None
        path = self.vowel_table_path
        if self._table is not None:
            t = self._table
            table_info = {"path": str(path), "file_sha1": t.meta.get("file_sha1"), "ko": t.ko, "en": t.en,
                          "max_distance": t.max_distance, "k": t.meta.get("k"), "seed": t.meta.get("seed"),
                          "check_u": t.meta.get("check_u")}
        elif path.is_file():
            table_info = {"path": str(path), "file_sha1": file_sha1(path), "loaded": False}
        return {
            "name": self.name,
            "frame_sec": self.frame_sec,
            "device": self.device,
            "tts_dir": str(self.tts_dir),
            "config": str(self.config_path),
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha1": self.checkpoint_sha1,
            "path_changes": {k: [str(a), str(b)] for k, (a, b) in self.path_changes.items()},
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "speaker": spk,
            "languages": {"en": cfg.tts_lang_en, "ko": cfg.tts_lang_ko,
                          "ids": dict(model.language_manager.name_to_id)},
            "noise_scale_dp": float(model.inference_noise_scale_dp),
            "length_scale": float(model.length_scale),
            "use_sdp": bool(model.args.use_sdp),
            "condition_dp_on_speaker": bool(model.args.condition_dp_on_speaker),
            "vocab_size": int(model.tokenizer.characters.num_chars),
            "blank_id": self.blank_id,
            "vowel_table": table_info,
            "missing_vowels": {k: dict(v) for k, v in self.missing_vowels.items()},
            "versions": {"coqui-tts": md.version("coqui-tts"), "torch": torch.__version__},
            "g2p": {**g2p.settings(self.en_voice, self.en_stress), "versions": g2p.versions(),
                    "trained_with": {k: self.g2p_settings.get(k) for k in ("scheme", "en", "versions")}},
        }

    def close(self) -> None:
        """모델을 내린다."""
        cuda = str(self.device).startswith("cuda")
        self.model = None
        self._speaker = {"kind": "none", "tensor": None}
        self._lang_ids = {"en": None, "ko": None}
        gc.collect()
        if cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
