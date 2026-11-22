"""SentencePiece 기반 한국어 서브워드 토크나이저 학습·로드 및 코퍼스 추출 모듈."""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Set, Tuple

import sentencepiece as spm

TEXT_UNK_ID = 0
TEXT_BOS_ID = 1
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3

SPECIAL_TOKENS: Tuple[str, str, str, str] = ("<unk>", "<s>", "</s>", "<pad>")
SPACE_MARKER = "\u2581"

_EMPTY_TEXT_MARKERS = {"", "-", "n/a", "na", "null", "none", "x"}
_LEADING_TRANSCRIPT_NOISE_RE = re.compile(r"^(?:##+|\*\*+|//+)\s*")
_MAX_VOCAB_HINT_RE = re.compile(r"<=\s*(\d+)")
_MIN_VOCAB_HINT_RE = re.compile(r"smaller than required_chars\.\s*\d+\s*vs\s*(\d+)")


class TokenizerProtocol(Protocol):
    """EDarch 파이프라인에서 사용하는 SentencePiece 호환 토크나이저 인터페이스 프로토콜입니다."""

    def EncodeAsIds(self, text: str) -> List[int]:
        """문자열을 서브워드 토큰 ID 리스트로 인코딩합니다."""
        ...

    def EncodeAsPieces(self, text: str) -> List[str]:
        """문자열을 서브워드 문자열 조각 리스트로 인코딩합니다."""
        ...

    def DecodeIds(self, ids: Sequence[int]) -> str:
        """토큰 ID 시퀀스를 원본 문자열로 디코딩합니다."""
        ...

    def GetPieceSize(self) -> int:
        """전체 어휘 사전 크기를 반환합니다."""
        ...

    def IdToPiece(self, idx: int) -> str:
        """토큰 ID에 대응하는 서브워드 문자열을 반환합니다."""
        ...

    def PieceToId(self, piece: str) -> int:
        """서브워드 문자열에 대응하는 토큰 ID를 반환합니다."""
        ...

    def unk_id(self) -> int:
        """미등록(UNK) 특수 토큰 ID를 반환합니다."""
        ...

    def bos_id(self) -> int:
        """문장 시작(BOS) 특수 토큰 ID를 반환합니다."""
        ...

    def eos_id(self) -> int:
        """문장 종료(EOS) 특수 토큰 ID를 반환합니다."""
        ...

    def pad_id(self) -> int:
        """패딩(PAD) 특수 토큰 ID를 반환합니다."""
        ...


def normalize_korean_text(text: str) -> str:
    """입력 텍스트를 한글 완성형(NFC)으로 정규화하고 연속 공백을 단일 공백으로 정리합니다."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text))
    return " ".join(norm.strip().split())


def resolve_spm_model_path(path: Path | str) -> Path:
    """출력 경로에 `.model` 확장자가 없으면 자동으로 부여하여 정규화된 경로를 반환합니다."""
    out_path = Path(path)
    if out_path.suffix.lower() == ".json":
        raise ValueError(
            f"커스텀 JSON 토크나이저 포맷은 지원되지 않습니다: {out_path}. "
            "SentencePiece 모델 파일 경로(예: ko_spm.model)를 지정하세요."
        )
    if out_path.suffix.lower() != ".model":
        return Path(f"{out_path}.model")
    return out_path


def load_korean_tokenizer(path: Path | str) -> TokenizerProtocol:
    """SentencePiece 바이너리 `.model` 파일을 로드하고 특수 토큰 규격을 검증합니다."""
    tok_path = Path(path)
    if not tok_path.exists():
        raise FileNotFoundError(f"SentencePiece 토크나이저 파일을 찾을 수 없습니다: {tok_path}")

    if tok_path.suffix.lower() == ".json":
        raise ValueError(
            f"{tok_path} 파일은 레거시 JSON 토크나이저입니다. "
            "SentencePiece 모델(.model) 파일을 지정하거나 새로 학습하세요."
        )
    try:
        with tok_path.open("rb") as f:
            head = f.read(16).lstrip()
        if head.startswith(b"{"):
            raise ValueError(
                f"{tok_path} 파일은 바이너리 SentencePiece 모델이 아닌 JSON 파일입니다. "
                "SentencePiece 모델(.model)을 사용하세요."
            )
    except OSError as exc:
        raise OSError(f"토크나이저 파일을 읽을 수 없습니다: {tok_path}") from exc

    proc = spm.SentencePieceProcessor()
    proc.Load(str(tok_path))
    return proc


def train_korean_sentencepiece(
    sentences: Sequence[str],
    output_model_path: Path | str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
    byte_fallback: bool = True,
) -> Path:
    """한국어 문장 코퍼스로부터 EDarch 특수 토큰 규격의 SentencePiece `.model`을 학습하고 저장 경로를 반환합니다."""
    min_required = max(TEXT_EOS_ID, TEXT_PAD_ID) + 8
    if vocab_size < min_required:
        raise ValueError(f"vocab_size({vocab_size})는 최소 {min_required} 이상이어야 합니다.")
    if model_type not in ("unigram", "bpe", "char", "word"):
        raise ValueError(f"지원하지 않는 SentencePiece model_type입니다: {model_type}")

    cleaned_lines: List[str] = []
    for line in sentences:
        norm_line = normalize_korean_text(line)
        if norm_line:
            cleaned_lines.append(norm_line)
    if not cleaned_lines:
        raise ValueError("SentencePiece 학습을 위한 유효한 한국어 문장이 비어 있습니다.")

    final_model_path = resolve_spm_model_path(output_model_path).resolve()
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = str(final_model_path.with_suffix(""))

    use_byte_fb = bool(byte_fallback and vocab_size >= 300)
    unique_chars = len({ch for s in cleaned_lines for ch in s if not ch.isspace()})
    base_pieces = 260 if use_byte_fb else 4
    min_needed_vocab = base_pieces + unique_chars + 4

    if vocab_size < min_needed_vocab and use_byte_fb and vocab_size < 512:
        use_byte_fb = False
        base_pieces = 4
        min_needed_vocab = base_pieces + unique_chars + 4

    corpus_upper = base_pieces + max(32, unique_chars * 4, len(cleaned_lines) * 4)
    cur_vocab = min(vocab_size, max(min_needed_vocab, corpus_upper))
    cur_vocab = max(min_required, min(vocab_size, cur_vocab))

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as tmp_f:
        tmp_corpus_path = Path(tmp_f.name)
        for line in cleaned_lines:
            tmp_f.write(line + "\n")

    try:
        for _ in range(12):
            try:
                spm.SentencePieceTrainer.Train(
                    input=str(tmp_corpus_path),
                    model_prefix=model_prefix,
                    vocab_size=cur_vocab,
                    character_coverage=character_coverage,
                    model_type=model_type,
                    byte_fallback=use_byte_fb,
                    unk_id=TEXT_UNK_ID,
                    bos_id=TEXT_BOS_ID,
                    eos_id=TEXT_EOS_ID,
                    pad_id=TEXT_PAD_ID,
                    hard_vocab_limit=False,
                    minloglevel=2,
                )
                break
            except RuntimeError as exc:
                msg = str(exc)
                if "Vocabulary size too high" in msg:
                    m_max = _MAX_VOCAB_HINT_RE.search(msg)
                    if m_max is not None:
                        hinted = int(m_max.group(1))
                        if hinted < cur_vocab and hinted >= min_required:
                            cur_vocab = hinted
                            continue
                    if cur_vocab > min_required:
                        cur_vocab = max(min_required, int(cur_vocab * 0.75))
                        continue
                if "smaller than required_chars" in msg:
                    m_min = _MIN_VOCAB_HINT_RE.search(msg)
                    needed = int(m_min.group(1)) if m_min is not None else (cur_vocab + 64)
                    if needed <= vocab_size and needed > cur_vocab:
                        cur_vocab = needed
                        continue
                    if use_byte_fb:
                        use_byte_fb = False
                        cur_vocab = min(vocab_size, max(min_required, unique_chars + 16))
                        continue
                raise
    finally:
        try:
            tmp_corpus_path.unlink(missing_ok=True)
        except OSError:
            pass

    proc = load_korean_tokenizer(final_model_path)
    if (
        proc.unk_id() != TEXT_UNK_ID
        or proc.bos_id() != TEXT_BOS_ID
        or proc.eos_id() != TEXT_EOS_ID
        or proc.pad_id() != TEXT_PAD_ID
    ):
        raise RuntimeError(
            f"학습된 SentencePiece 특수 토큰 ID가 규격(0,1,2,3)과 다릅니다: "
            f"unk={proc.unk_id()}, bos={proc.bos_id()}, eos={proc.eos_id()}, pad={proc.pad_id()}"
        )
    return final_model_path


def _clean_text_field(val: Any) -> str:
    """전사·번역 문자열 필드에서 노이즈 접두어를 제거하고 비어 있는 마커를 필터링합니다."""
    if val is None or isinstance(val, bool):
        return ""
    text = str(val).strip()
    text = _LEADING_TRANSCRIPT_NOISE_RE.sub("", text).strip()
    if text.lower() in _EMPTY_TEXT_MARKERS:
        return ""
    return text


def _normalize_record_keys(raw: dict[Any, Any]) -> Tuple[dict[str, Any], dict[str, Any]]:
    """JSON 레코드 키에서 BOM과 공백을 제거한 원본 키 사전 및 소문자 키 사전을 반환합니다."""
    norm: dict[str, Any] = {}
    lower_norm: dict[str, Any] = {}
    for k, v in raw.items():
        ck = str(k).lstrip("\ufeff").strip()
        norm[ck] = v
        lower_norm[ck.lower()] = v
    return norm, lower_norm


def _extract_json_records(payload: Any) -> List[dict[str, Any]]:
    """리스트 또는 딕셔너리 형태의 AI-Hub JSON 최상위 구조에서 레코드 목록을 추출합니다."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        norm, lower_norm = _normalize_record_keys(payload)
        for list_key in ("data", "utterances", "dialogue", "dialogs", "sentences"):
            cand = lower_norm.get(list_key)
            if isinstance(cand, list):
                return [x for x in cand if isinstance(x, dict)]
        return [norm]
    return []


def _classify_71379_accent(t_code_raw: str, en_wav_name: str, json_name: str) -> Optional[str]:
    """AI-Hub 71379 레코드의 언어 코드·파일명에서 미국식('us') 또는 영국식('uk') 여부를 판별합니다."""
    tc = t_code_raw.strip().lower()
    if tc in ("es", "es-es", "ru", "ru-ru"):
        return None
    if tc in ("en-gb", "uk", "en_gb"):
        return "uk"
    if tc in ("en-us", "en", "en_us", "eng"):
        return "us"

    lower_en = en_wav_name.lower()
    lower_jp = json_name.lower()
    for sfx in ("_es.wav", "_ru.wav", "_es.json", "_ru.json"):
        if lower_en.endswith(sfx) or lower_jp.endswith(sfx):
            return None
    if lower_en.endswith("_uk.wav") or lower_jp.endswith("_uk.json"):
        return "uk"
    if lower_en.endswith("_en.wav") or lower_jp.endswith("_en.json"):
        return "us"
    return None


def extract_korean_sentences_from_path(
    input_path: Path,
    use_mt_fallback: bool = False,
    en_accent_71379: str = "all",
) -> List[str]:
    """텍스트 파일(`.txt`) 또는 AI-Hub 71693·71379 JSON 파일/디렉터리에서 한국어 문장을 추출합니다."""
    if not input_path.exists():
        raise FileNotFoundError(f"토크나이저 학습 입력 경로를 찾을 수 없습니다: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() != ".json":
        lines = input_path.read_text(encoding="utf-8-sig").splitlines()
        return [normalize_korean_text(ln) for ln in lines if normalize_korean_text(ln)]

    json_files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.json"))
    stt_text_by_stem: dict[str, str] = {}
    stt_ordered_texts: List[Tuple[str, str]] = []
    deidentified_stems: Set[str] = set()
    trans_71379_records: List[Tuple[str, dict[str, Any]]] = []
    direct_sentences: List[str] = []

    for jf in json_files:
        lower_jname = jf.name.lower()
        if lower_jname.endswith(("_es.json", "_ru.json")):
            continue
        try:
            payload = json.loads(jf.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue

        for raw_rec in _extract_json_records(payload):
            _, item = _normalize_record_keys(raw_rec)

            if (
                "source_language" in item
                and "target_language" in item
                and ("audio_start" in item or "mtpe" in item or "mt" in item)
            ):
                src_lang = str(item.get("source_language") or "").strip().lower()
                tgt_lang = str(item.get("target_language") or "").strip().lower()
                if src_lang.startswith("en") and tgt_lang.startswith("ko"):
                    ko_txt = _clean_text_field(item.get("mtpe"))
                    if not ko_txt and use_mt_fallback:
                        ko_txt = _clean_text_field(item.get("mt"))
                    if ko_txt:
                        direct_sentences.append(ko_txt)
                continue

            if "음성파일명" in item and "t-fname" not in item and "t-code" not in item:
                ko_fname = _clean_text_field(item.get("음성파일명"))
                stem_key = Path(ko_fname).stem.lower() if ko_fname else ""
                is_deid = str(item.get("비식별화") or "").strip().upper() == "O"
                if is_deid:
                    if stem_key:
                        deidentified_stems.add(stem_key)
                    continue
                stt_txt = (
                    _clean_text_field(item.get("번역본"))
                    or _clean_text_field(item.get("철자전사"))
                    or _clean_text_field(item.get("발음전사"))
                    or _clean_text_field(item.get("전사작업본"))
                )
                if stt_txt:
                    if stem_key:
                        stt_text_by_stem[stem_key] = stt_txt
                    stt_ordered_texts.append((stem_key, stt_txt))
                continue

            if any(k in item for k in ("t-fname", "s-fname", "t-code", "최종번역문", "최종번역")):
                trans_71379_records.append((jf.name, item))
                continue

            generic_txt = (
                _clean_text_field(item.get("ko_mtpe"))
                or _clean_text_field(item.get("ko"))
                or _clean_text_field(item.get("한국어"))
                or _clean_text_field(item.get("수정원문"))
                or _clean_text_field(item.get("원문"))
            )
            if not generic_txt and use_mt_fallback:
                generic_txt = _clean_text_field(item.get("ko_mt"))
            if not generic_txt:
                label_info = item.get("라벨링") or item.get("labeling")
                if isinstance(label_info, dict):
                    _, lower_lbl = _normalize_record_keys(label_info)
                    generic_txt = _clean_text_field(lower_lbl.get("한국어")) or _clean_text_field(
                        lower_lbl.get("ko")
                    )
            if generic_txt:
                direct_sentences.append(generic_txt)

    for jname, item in trans_71379_records:
        s_code = str(item.get("s-code") or "ko-KR").strip().lower()
        if s_code and not s_code.startswith("ko"):
            continue
        ko_wav_name = _clean_text_field(item.get("s-fname"))
        en_wav_name = _clean_text_field(item.get("t-fname"))
        t_code = str(item.get("t-code") or "").strip()
        variant = _classify_71379_accent(t_code, en_wav_name, jname)
        if variant is None:
            continue
        if en_accent_71379 != "all" and variant != en_accent_71379:
            continue
        ko_stem = Path(ko_wav_name).stem.lower() if ko_wav_name else ""
        if ko_stem and ko_stem in deidentified_stems:
            continue
        if str(item.get("비식별화") or "").strip().upper() == "O":
            continue
        ko_txt = _clean_text_field(item.get("수정원문")) or _clean_text_field(item.get("원문"))
        if not ko_txt and ko_stem:
            ko_txt = stt_text_by_stem.get(ko_stem, "")
        if ko_txt:
            direct_sentences.append(ko_txt)

    if not trans_71379_records and stt_ordered_texts:
        for stem_key, stt_txt in stt_ordered_texts:
            if not stem_key or stem_key not in deidentified_stems:
                direct_sentences.append(stt_txt)

    sentences: List[str] = []
    seen: Set[str] = set()
    for raw_sent in direct_sentences:
        norm_sent = normalize_korean_text(raw_sent)
        if norm_sent and norm_sent not in seen:
            seen.add(norm_sent)
            sentences.append(norm_sent)
    return sentences


def train_spm_tokenizer_from_cli(
    input_path: Path,
    output_path: Path,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    byte_fallback: bool = True,
    use_mt_fallback: bool = False,
) -> Path:
    """CLI 입력 경로에서 한국어 문장을 추출하여 SentencePiece(`.model`) 토크나이저를 학습·저장합니다."""
    sentences = extract_korean_sentences_from_path(
        input_path=input_path,
        use_mt_fallback=use_mt_fallback,
    )
    if not sentences:
        raise ValueError(f"입력 경로에서 유효한 한국어 문장을 찾지 못했습니다: {input_path}")
    saved_path = train_korean_sentencepiece(
        sentences=sentences,
        output_model_path=output_path,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        byte_fallback=byte_fallback,
    )
    proc = load_korean_tokenizer(saved_path)
    print(
        f"[Done] SentencePiece 토크나이저 저장 완료: {saved_path} "
        f"(문장 수={len(sentences)}, vocab_size={proc.GetPieceSize()}, model_type={model_type})"
    )
    return saved_path


def run_tokenizer_self_test() -> None:
    """한국어 SentencePiece 토크나이저의 학습·특수 토큰 규격·바이트 폴백·무손실 복원을 검증합니다."""
    sample_corpus = [
        "안녕하세요, 국제 학술대회 동시통역 시스템 테스트입니다.",
        "인공지능 기반 실시간 음성 번역 모델은 낮은 지연시간과 높은 정확도를 목표로 합니다.",
        "영어 발화가 끝나기 전에 한국어 번역 토큰과 음성을 스트리밍으로 생성합니다.",
        "실시간 동시통역에서는 문맥을 파악하며 자연스러운 한국어 어순으로 번역해야 합니다.",
        "강화학습을 통해 지연시간과 번역 품질 사이의 균형을 스스로 최적화합니다.",
        "방송 콘텐츠와 강연 음성의 다양한 화자 특성을 반영하여 자연스럽게 발화합니다.",
    ] * 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = train_korean_sentencepiece(
            sentences=sample_corpus,
            output_model_path=Path(tmp_dir) / "self_test_ko.model",
            vocab_size=400,
            model_type="unigram",
            byte_fallback=True,
        )
        tok = load_korean_tokenizer(model_path)

        test_sentences = [
            "안녕하세요, 실시간 동시통역 테스트입니다!",
            "미등록 한자(韓英)와 이모지(🚀) 및 숫자 2026도 무손실 복원됩니다.",
        ]

        print(
            f"[Self-Test] SentencePiece 학습·로드 완료 "
            f"(vocab_size={tok.GetPieceSize()}, path={model_path.name})"
        )
        assert (
            tok.unk_id() == TEXT_UNK_ID
            and tok.bos_id() == TEXT_BOS_ID
            and tok.eos_id() == TEXT_EOS_ID
            and tok.pad_id() == TEXT_PAD_ID
        )

        for sent in test_sentences:
            ids = list(tok.EncodeAsIds(sent))
            pieces = list(tok.EncodeAsPieces(sent))
            decoded = tok.DecodeIds(ids)
            assert TEXT_UNK_ID not in ids
            assert all(tid not in (TEXT_EOS_ID, TEXT_PAD_ID) for tid in ids)
            assert normalize_korean_text(decoded) == normalize_korean_text(sent)
            print(f"  - 원문: {sent}")
            print(
                f"    서브워드({len(pieces)}개): {pieces[:12]}"
                f"{' ...' if len(pieces) > 12 else ''}"
            )
            print(f"    복원: {decoded}")

    print("[OK] 한국어 SentencePiece 토크나이저 무결성 검증 통과!")
