"""scripts/yourtts 학습 레시피: 8:1:1 나누기, 매니페스트(LibriTTS 구조, CSV), phonemes 칸과 characters.txt, d-vector 파일 꼴,
학습 표본, 설정값이 논문 4.1절 값·공개 YourTTS 구조와 같은지, 소리 읽기(Windows 안전), 걸음 수 제한, 층 가져오기."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from pstts.config import ROOT
from pstts.paper import g2p

sys.path.insert(0, str(ROOT / "scripts" / "yourtts"))
import compute_dvectors  # noqa: E402
import make_manifest  # noqa: E402
import phonemize_manifest  # noqa: E402
import train_yourtts  # noqa: E402

from TTS.tts.models.vits import Vits  # noqa: E402
from TTS.tts.utils.speakers import SpeakerManager  # noqa: E402

PUBLIC_CONFIG = ROOT / "models" / "your_tts" / "tts_models--multilingual--multi-dataset--your_tts" / "config.json"


def _wav(path: Path, sec: float = 0.3, sr: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sec * sr)) / sr
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------- 8:1:1
@pytest.mark.parametrize("n,want", [(8, (6, 1, 1)), (10, (8, 1, 1)), (16, (13, 1, 2)), (1000, (800, 100, 100)),
                                    (5, (4, 1, 0)), (1, (1, 0, 0))])
def test_split_811_sizes(n: int, want: tuple[int, int, int]) -> None:
    rows = [{"i": str(i)} for i in range(n)]
    parts = make_manifest.split_811(rows, seed=0)
    assert tuple(len(p) for p in parts) == want
    assert sorted(r["i"] for p in parts for r in p) == sorted(r["i"] for r in rows)


def test_split_811_is_seeded() -> None:
    rows = [{"i": str(i)} for i in range(50)]
    a = make_manifest.split_811(rows, 0)
    assert a == make_manifest.split_811(rows, 0)
    assert a != make_manifest.split_811(rows, 1)
    assert rows == [{"i": str(i)} for i in range(50)]         # 넘긴 목록은 바꾸지 않는다


# ---------------------------------------------------------------------------- 매니페스트
def _libritts(root: Path) -> None:
    for spk in ("14", "16", "17"):
        for chap in ("100", "200"):
            for k in range(2):
                stem = f"{spk}_{chap}_000{k}_000000"
                wav = _wav(root / "train-clean-360" / spk / chap / f"{stem}.wav")
                if spk == "17" and chap == "200" and k == 1:
                    continue                                     # 글 파일이 없는 발화
                wav.with_suffix(".normalized.txt").write_text(f"Speaker {spk} says line {k}.\n", encoding="utf-8")


def test_libritts_rows(tmp_path: Path) -> None:
    _libritts(tmp_path / "LibriTTS")
    rows, skipped = make_manifest.libritts_rows(tmp_path / "LibriTTS")
    assert len(rows) == 11 and skipped == {"no_text": 1}
    assert {r["speaker"] for r in rows} == {"libritts_14", "libritts_16", "libritts_17"}
    assert {r["lang"] for r in rows} == {"en"}
    assert all(Path(r["audio"]).is_absolute() and Path(r["audio"]).is_file() for r in rows)
    assert rows[0]["text"] == "Speaker 14 says line 0."
    with pytest.raises(FileNotFoundError):
        make_manifest.libritts_rows(tmp_path / "nope")


def _csv(path: Path, rows: list[dict[str, str]], with_lang: bool = True) -> Path:
    cols = ["audio", "text", "speaker"] + (["lang"] if with_lang else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return path


def test_make_manifest_splits_each_dataset_811(tmp_path: Path) -> None:
    _libritts(tmp_path / "LibriTTS")
    ko_dir = tmp_path / "kmsss"
    rows = [{"audio": f"wav/k{i}.wav", "text": f"안녕하세요 {i}번", "speaker": f"F{i % 3}", "lang": "ko"} for i in range(10)]
    for r in rows:
        _wav(ko_dir / r["audio"])
    rows.append({"audio": "wav/missing.wav", "text": "없음", "speaker": "F0", "lang": "ko"})
    kcsv = _csv(ko_dir / "kmsss.csv", rows, with_lang=False)
    out = tmp_path / "man"
    assert make_manifest.main(["--libritts-dir", str(tmp_path / "LibriTTS"), "--csv", str(kcsv), "--csv-lang", "ko",
                               "--out", str(out), "--seed", "0"]) == 0
    parts = {s: make_manifest.read_manifest(make_manifest.manifest_path(out, s)) for s in make_manifest.SPLITS}
    info = json.loads((out / "manifest_info.json").read_text(encoding="utf-8"))
    assert info["datasets"]["libritts"] == {"rows": 11, "train": 9, "valid": 1, "test": 1, "speakers": 3,
                                            "langs": {"en": 11}, "skipped": {"no_text": 1}}
    assert info["datasets"]["kmsss"]["train"] == 8 and info["datasets"]["kmsss"]["valid"] == 1
    assert info["datasets"]["kmsss"]["skipped"] == {"no_audio": 1}
    assert [len(parts[s]) for s in make_manifest.SPLITS] == [17, 2, 2]
    header = (out / "manifest_train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "audio,text,speaker,lang"
    ko = [r for p in parts.values() for r in p if r["lang"] == "ko"]
    assert len(ko) == 10 and all(r["speaker"].startswith("kmsss_F") for r in ko)
    assert all(Path(r["audio"]).is_absolute() for p in parts.values() for r in p)
    keys = [make_manifest.sample_key(r) for p in parts.values() for r in p]
    assert len(set(keys)) == len(keys) == 21
    assert keys[0].split("#", 1)[0] in ("en", "ko")


def test_make_manifest_rejects_bad_lang_and_duplicates(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav")
    bad = _csv(tmp_path / "bad.csv", [{"audio": "a.wav", "text": "x", "speaker": "s", "lang": "fr"}])
    with pytest.raises(ValueError):
        make_manifest.main(["--csv", str(bad), "--out", str(tmp_path / "o")])
    row = {"audio": "a.wav", "text": "x", "speaker": "s", "lang": "en"}
    one = _csv(tmp_path / "one.csv", [row])
    two = _csv(tmp_path / "two.csv", [row])
    with pytest.raises(ValueError, match="두 번"):
        make_manifest.main(["--csv", str(one), "--csv", str(two), "--out", str(tmp_path / "o")])


# ---------------------------------------------------------------------------- 음소 칸
def _manifest_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    rows = [{"audio": f"e{i}.wav", "text": t, "speaker": "a", "lang": "en"}
            for i, t in enumerate(["We have to leave before the storm hits.", "Hello, world!", "I see.",
                                   "Two tickets, please.", "Go now.", "It's raining."])]
    rows += [{"audio": f"k{i}.wav", "text": t, "speaker": "b", "lang": "ko"}
             for i, t in enumerate(["같이 읽고 신라에 가자.", "폭풍이 오기 전에 떠나야 해.", "의사 회의 희망", "좋아요.",
                                    "고마워!", "어디 가?"])]
    for r in rows:
        _wav(data / r["audio"])
    out = tmp_path / "man"
    make_manifest.main(["--csv", str(_csv(data / "mix.csv", rows)), "--out", str(out)])
    return out


def test_phonemize_manifest_column_and_characters(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    assert phonemize_manifest.main(["--manifest-dir", str(man)]) == 0
    seen: set[str] = set()
    n = 0
    for split in make_manifest.SPLITS:
        lines = (man / f"manifest_{split}.csv").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "audio,text,speaker,lang,phonemes"
        for row in make_manifest.read_manifest(make_manifest.manifest_path(man, split)):
            want = g2p.phonemes(row["text"], row["lang"])
            assert row["phonemes"] == "".join(want) and list(row["phonemes"]) == want     # 빈 토큰 없이 이어 붙인 글
            assert g2p.tokens(row["text"], row["lang"]) == g2p.with_blanks(list(row["phonemes"]))
            seen.update(row["phonemes"])
            n += 1
    assert n == 12
    chars = phonemize_manifest.read_characters(man / "characters.txt")
    assert chars == sorted(seen, key=ord)
    assert " " in chars and "ㅏ" in chars and "ㅇ" in chars and "ˈ" not in chars and "ː" in chars
    assert not any(0x1100 <= ord(c) <= 0x11FF for c in chars)              # 조합형 자모 없음(Fig. 4 호환 자모)
    text = (man / "characters.txt").read_text(encoding="utf-8")
    assert "U+0020\t\t" in text and "ko=vowel" in text and "en=mark" in text
    info = json.loads((man / "phonemize_info.json").read_text(encoding="utf-8"))
    assert info["characters"]["count"] == len(chars) and info["dropped_empty"] == {"train": 0, "valid": 0, "test": 0}
    settings = g2p.read_settings(man / g2p.SETTINGS_NAME)                   # 학습 폴더로 복사돼 추론이 확인한다
    g2p.check_settings(settings)
    assert settings["en"]["voice"] == "en-gb" and settings["en"]["with_stress"] is False
    assert info["g2p"]["scheme"] == g2p.SCHEME


def test_phonemize_manifest_en_voice_options(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    assert phonemize_manifest.main(["--manifest-dir", str(man), "--en-voice", "en-us", "--en-stress"]) == 0
    rows = [r for s in make_manifest.SPLITS for r in make_manifest.read_manifest(make_manifest.manifest_path(man, s))]
    for r in rows:
        assert r["phonemes"] == "".join(g2p.phonemes(r["text"], r["lang"], en_voice="en-us", en_stress=True))
    assert "ˈ" in phonemize_manifest.read_characters(man / "characters.txt")
    settings = g2p.read_settings(man / g2p.SETTINGS_NAME)
    assert settings["en"]["voice"] == "en-us" and settings["en"]["with_stress"] is True
    with pytest.raises(ValueError, match="--g2p-en-voice en-us --g2p-en-stress"):
        g2p.check_settings(settings)                                       # 기본 설정으로 추론하면 멈춘다


def test_phonemize_extra_text_only_adds_characters(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    extra = tmp_path / "extra_ko.txt"
    extra.write_text("쿼츠 퓨즈 뷁\n\n", encoding="utf-8")
    assert phonemize_manifest.main(["--manifest-dir", str(man), "--extra-ko", str(extra)]) == 0
    rows = [r for s in make_manifest.SPLITS for r in make_manifest.read_manifest(make_manifest.manifest_path(man, s))]
    assert len(rows) == 12                                            # 학습 줄은 그대로
    trained = {t for r in rows for t in r["phonemes"]}
    chars = phonemize_manifest.read_characters(man / "characters.txt")
    added = set(chars) - trained
    assert added and added <= set(g2p.phonemes("쿼츠 퓨즈 뷁", "ko"))
    info = json.loads((man / "phonemize_info.json").read_text(encoding="utf-8"))
    assert info["characters"]["only_from_extra"] == len(added) and info["extra_sentences"] == {"ko": 1}
    assert "extra_ko=" in (man / "characters.txt").read_text(encoding="utf-8")


def test_read_characters_rejects_bad_files(tmp_path: Path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("U+0061\ta\nU+0061\ta\n", encoding="utf-8")
    with pytest.raises(ValueError):
        phonemize_manifest.read_characters(p)
    p.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        phonemize_manifest.read_characters(p)
    p.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError):
        phonemize_manifest.read_characters(p)


# ---------------------------------------------------------------------------- d-vector 파일
class FakeEncoderManager:
    """compute_embedding_from_clip 만 흉내 낸다(진짜 화자 인코더는 스모크에서 돈다)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def compute_embedding_from_clip(self, path: str) -> list[float]:
        self.calls.append(path)
        rng = np.random.default_rng(len(self.calls))
        v = rng.standard_normal(512)
        return (v / np.linalg.norm(v)).tolist()


def test_compute_dvectors_writes_coqui_d_vector_file(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    rows = [r for s in make_manifest.SPLITS for r in make_manifest.read_manifest(make_manifest.manifest_path(man, s))]
    fake = FakeEncoderManager()
    out = man / "d_vectors.pth"
    s = compute_dvectors.compute(rows, fake, out, base=man, save_every=5)
    assert s["vectors"] == s["computed"] == 12 and s["dims"] == [512] and s["speakers"] == 2
    sm = SpeakerManager(d_vectors_file_path=str(out))                # Coqui 가 그대로 읽는 꼴
    assert sorted(sm.embedding_names) == ["mix_a", "mix_b"]
    assert set(sm.embeddings) == {make_manifest.sample_key(r) for r in rows}
    again = compute_dvectors.compute(rows, fake, out, base=man)      # 이어하기: 이미 있는 키는 건너뛴다
    assert again["computed"] == 0 and again["reused"] == 12 and len(fake.calls) == 12


def test_relative_audio_paths_resolve_like_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """손으로 만든 매니페스트의 상대 소리 경로는 d-vector 와 학습 모두 매니페스트 폴더 기준이다. 키는 적힌 글 그대로."""
    man = tmp_path / "man"
    rows = [{"audio": f"wav/r{i}.wav", "text": "Go now.", "speaker": "s", "lang": "en"} for i in range(3)]
    for r in rows:
        _wav(man / r["audio"])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                                       # 지금 폴더 기준이면 못 찾는다
    fake = FakeEncoderManager()
    compute_dvectors.compute(rows, fake, man / "d.pth", base=man)
    assert fake.calls == [str(man / r["audio"]) for r in rows] and all(Path(c).is_file() for c in fake.calls)
    keys = set(SpeakerManager(d_vectors_file_path=str(man / "d.pth")).embeddings)
    assert keys == {make_manifest.sample_key(r) for r in rows} == {f"en#wav/r{i}.wav" for i in range(3)}
    for r in rows:
        r["phonemes"] = "".join(g2p.phonemes(r["text"], r["lang"]))
    make_manifest.write_manifest(man / "manifest_train.csv", rows, (*make_manifest.MANIFEST_COLUMNS, "phonemes"))
    samples, _ = train_yourtts.load_samples(man / "manifest_train.csv")
    assert [s["audio_file"] for s in samples] == fake.calls
    assert [s["audio_unique_name"] for s in samples] == [make_manifest.sample_key(r) for r in rows]


# ---------------------------------------------------------------------------- 소리 꼴 확인
def test_audio_problem_matches_coqui_formats(tmp_path: Path) -> None:
    assert make_manifest.audio_problem(_wav(tmp_path / "ok.wav")) is None
    flac = tmp_path / "ok.flac"
    sf.write(flac, np.zeros(2205, dtype=np.float32) + 0.1, 22050, format="FLAC")
    assert make_manifest.audio_problem(flac) is None
    ogg = tmp_path / "x.ogg"
    sf.write(ogg, np.zeros(2205, dtype=np.float32) + 0.1, 22050, format="OGG")
    assert "확장자 .ogg" in make_manifest.audio_problem(ogg)               # soundfile 은 읽지만 Coqui 가 뺀다
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav file")
    assert "읽지 못함" in make_manifest.audio_problem(broken)
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), 22050, subtype="PCM_16")
    assert "0" in make_manifest.audio_problem(empty)


def test_csv_rows_stop_on_audio_coqui_would_drop(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav")
    sf.write(tmp_path / "b.ogg", np.zeros(2205, dtype=np.float32) + 0.1, 22050, format="OGG")
    (tmp_path / "c.pcm").write_bytes(b"\x00\x01" * 100)
    rows = [{"audio": n, "text": "x", "speaker": "s", "lang": "ko"} for n in ("a.wav", "b.ogg", "c.pcm")]
    path = _csv(tmp_path / "k.csv", rows)
    with pytest.raises(ValueError, match="학습에 쓸 수 없는 소리 2개") as e:
        make_manifest.csv_rows(path, "ko", "k")
    assert "b.ogg" in str(e.value) and "c.pcm" in str(e.value) and "a.wav" not in str(e.value)
    ok, _ = make_manifest.csv_rows(_csv(tmp_path / "ok.csv", rows[:1]), "ko", "ok")
    assert len(ok) == 1


def test_libritts_rows_stop_on_unreadable_wav(tmp_path: Path) -> None:
    _libritts(tmp_path / "LibriTTS")
    bad = tmp_path / "LibriTTS" / "train-clean-360" / "14" / "100" / "14_100_0009_000000.wav"
    bad.write_bytes(b"broken")
    bad.with_suffix(".normalized.txt").write_text("Broken line.", encoding="utf-8")
    with pytest.raises(ValueError, match="14_100_0009_000000.wav"):
        make_manifest.libritts_rows(tmp_path / "LibriTTS")


# ---------------------------------------------------------------------------- 설정값
def _dvec_file(tmp_path: Path) -> Path:
    """Vits 를 만들 때 SpeakerManager 가 읽는 작은 d-vector 파일."""
    from TTS.tts.utils.managers import save_file

    p = tmp_path / "d.pth"
    if not p.is_file():
        save_file({"en#x.wav": {"name": "s", "embedding": [0.0] * 511 + [1.0]}}, str(p))
    return p


def _config(tmp_path: Path, **kw) -> object:
    return train_yourtts.build_config(characters=list("abc ˈ"), d_vector_file=str(_dvec_file(tmp_path)),
                                      output_path=str(tmp_path), **kw)


def test_config_has_paper_values(tmp_path: Path) -> None:
    c = _config(tmp_path)
    # 논문 4.1절
    assert c.optimizer == "AdamW"
    assert c.optimizer_params["betas"] == [0.8, 0.99] and c.optimizer_params["weight_decay"] == 0.01
    assert c.lr_gen == c.lr_disc == 2e-4
    assert c.epochs == 200 and c.batch_size == 48
    # Your-TTS 구조
    a = c.model_args
    assert a.use_sdp and a.condition_dp_on_speaker
    assert a.use_language_embedding and a.embedded_language_dim == 4 and c.languages == ["en", "ko"]
    assert a.use_d_vector_file and a.d_vector_dim == 512 and not a.use_speaker_embedding
    assert c.use_d_vector_file and c.d_vector_dim == 512
    assert not a.use_speaker_encoder_as_loss
    assert c.add_blank and not c.use_phonemes and c.text_cleaner is None and not c.enable_eos_bos_chars
    # 소리(논문 3.1절 프레임: 22.05 kHz, hop 256, 창 1024)
    au = c.audio
    assert (au.sample_rate, au.hop_length, au.win_length, au.fft_size, au.num_mels) == (22050, 256, 1024, 1024, 80)
    assert c.characters.characters == "abc ˈ" and c.characters.characters_class.endswith("VitsCharacters")
    assert c.use_weighted_sampler and c.weighted_sampler_attrs == {"language": 1.0}
    assert c.test_sentences == [] and c.datasets == []


def test_yourtts_arch_matches_public_yourtts(tmp_path: Path) -> None:
    public = json.loads(PUBLIC_CONFIG.read_text(encoding="utf-8"))["model_args"]
    for key, val in train_yourtts.YOURTTS_ARCH.items():
        if key in public:
            assert public[key] == val, key
    c = _config(tmp_path)
    for key, val in train_yourtts.YOURTTS_ARCH.items():
        assert c.model_args[key] == val, key


def test_config_options_and_tokenizer(tmp_path: Path) -> None:
    c = _config(tmp_path, arch="tiny", batch_size=16, epochs=3, language_balance=False, num_workers=0)
    assert c.batch_size == c.eval_batch_size == 16 and c.epochs == 3 and not c.use_weighted_sampler
    assert c.model_args.hidden_channels == 16 and c.num_loader_workers == 0
    model = Vits(c)
    vocab = model.tokenizer.characters.vocab
    assert vocab[0] == "<PAD>" and vocab[-1] == "<BLNK>" and vocab[1:-1] == list("abc ˈ")
    assert model.tokenizer.text_to_ids("ab", "en") == [6, 1, 6, 2, 6]
    with pytest.raises(ValueError):
        _config(tmp_path, arch="huge")
    with pytest.raises(ValueError):
        train_yourtts.build_config(characters=["ab"], d_vector_file="d", output_path=str(tmp_path))


# ---------------------------------------------------------------------------- 학습 표본과 소리 읽기
def test_load_samples(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    with pytest.raises(ValueError, match="phonemes"):
        train_yourtts.load_samples(make_manifest.manifest_path(man, "train"))
    phonemize_manifest.main(["--manifest-dir", str(man)])
    path = make_manifest.manifest_path(man, "train")
    rows = make_manifest.read_manifest(path)
    samples, info = train_yourtts.load_samples(path)
    assert info == {"rows": len(rows), "used": len(rows), "skipped": {}}
    for s, r in zip(samples, rows):
        assert s["text"] == r["phonemes"] and s["language"] == r["lang"] and s["speaker_name"] == r["speaker"]
        assert s["audio_unique_name"] == make_manifest.sample_key(r) and s["audio_file"] == r["audio"]
    _, short = train_yourtts.load_samples(path, max_audio_sec=0.1)
    assert short["used"] == 0 and short["skipped"] == {"too_long": len(rows)}


def test_load_wav_tensor_resamples_without_torchaudio(tmp_path: Path) -> None:
    same = _wav(tmp_path / "a.wav", sec=0.5, sr=22050)
    x = train_yourtts.load_wav_tensor(str(same), 22050)
    ref, _ = sf.read(same, dtype="float32")
    assert x.shape == (1, ref.size) and x.dtype == torch.float32
    np.testing.assert_allclose(x[0].numpy(), ref, atol=1e-6)
    other = _wav(tmp_path / "b.wav", sec=0.5, sr=24000)
    y = train_yourtts.load_wav_tensor(str(other), 22050)
    assert abs(y.shape[1] - 11025) <= 1 and float(y.abs().max()) <= 1.0
    stereo = tmp_path / "c.wav"
    sf.write(stereo, np.stack([ref, ref], axis=1), 22050, subtype="PCM_16")
    assert train_yourtts.load_wav_tensor(str(stereo), 22050).shape == (1, ref.size)


def test_step_limited_loader_stops_at_max_steps() -> None:
    class T:
        max_steps = 5
        total_steps_done = 3

    t = T()
    loader = train_yourtts.StepLimitedLoader(list(range(10)), t)
    assert len(loader) == 10
    got = []
    for b in loader:
        got.append(b)
        t.total_steps_done += 1
    assert got == [0, 1]
    t.max_steps = None
    assert list(loader) == list(range(10))


def test_batch_limited_loader() -> None:
    loader = train_yourtts.BatchLimitedLoader(list(range(10)), 3)
    assert len(loader) == 3 and list(loader) == [0, 1, 2] and list(loader) == [0, 1, 2]
    assert len(train_yourtts.BatchLimitedLoader([1, 2], 5)) == 2


def test_max_steps_also_limits_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    """--max-steps 점검에서 검증도 한 번에 max_steps 배치만 읽는다(Coqui 는 검증 자료 전부를 읽음)."""
    seen: dict[str, object] = {}

    def fake_fit(self: train_yourtts.PsTrainer) -> None:
        seen["train"], seen["eval"] = self.train_loader, self.eval_loader

    monkeypatch.setattr(train_yourtts.Trainer, "fit", fake_fit)
    t = object.__new__(train_yourtts.PsTrainer)
    t.max_steps, t.total_steps_done, t.epochs_done = 4, 0, 0
    t.config = type("C", (), {"epochs": 200, "run_eval": True})()
    t.train_loader, t.eval_loader, t.eval_samples = list(range(10)), None, ["v"] * 100
    t.get_eval_dataloader = lambda samples, verbose=True: list(range(len(samples)))
    t.save_checkpoint = lambda: None
    train_yourtts.PsTrainer.fit(t)
    assert isinstance(seen["train"], train_yourtts.StepLimitedLoader)
    assert isinstance(seen["eval"], train_yourtts.BatchLimitedLoader) and list(seen["eval"]) == [0, 1, 2, 3]
    assert t.config.epochs == 200


def _sampler_samples(tmp_path: Path, n: int = 10) -> tuple[list[dict], list[str]]:
    """길이가 다른 소리(0.2~1.1초) n 개씩 두 언어. 학습 표본 꼴(phonemes 칸)과 글자 목록."""
    man = tmp_path / "sm"
    rows = []
    for lang, text in (("en", "Go now."), ("ko", "가자.")):
        for i in range(n):
            name = f"{lang}{i}.wav"
            _wav(man / name, sec=0.2 + 0.1 * i)
            rows.append({"audio": str(man / name), "text": text, "speaker": f"s_{lang}", "lang": lang,
                         "phonemes": "".join(g2p.phonemes(text, lang))})
    make_manifest.write_manifest(man / "manifest_train.csv", rows, (*make_manifest.MANIFEST_COLUMNS, "phonemes"))
    samples, _ = train_yourtts.load_samples(man / "manifest_train.csv")
    chars = sorted({c for r in rows for c in r["phonemes"]}, key=ord)
    return samples, chars


def _ps_vits(tmp_path: Path, chars: list[str], **kw) -> tuple[object, object]:
    config = train_yourtts.build_config(characters=chars, d_vector_file=str(_dvec_file(tmp_path)),
                                        output_path=str(tmp_path), arch="tiny", num_workers=0, **kw)
    return train_yourtts.PsVits(config), config


def _batches(loader: object) -> list[list[int]]:
    return [list(b) for b in loader.batch_sampler]


def test_no_language_balance_reshuffles_every_epoch(tmp_path: Path) -> None:
    samples, chars = _sampler_samples(tmp_path)
    model, config = _ps_vits(tmp_path, chars, batch_size=4, language_balance=False)
    loader = model.get_data_loader(config, is_eval=False, samples=samples, verbose=False)
    assert isinstance(loader.batch_sampler, train_yourtts.BucketBatchSampler)
    torch.manual_seed(0)
    first, second = _batches(loader), _batches(loader)
    assert first != second                                             # epoch 마다 다른 순서(Coqui 그대로면 같음)
    for b in (first, second):
        assert len(b) == 5 and sorted(i for batch in b for i in batch) == list(range(20))


def test_start_by_longest_keeps_sorted_order_for_memory_probe(tmp_path: Path) -> None:
    samples, chars = _sampler_samples(tmp_path)
    model, config = _ps_vits(tmp_path, chars, batch_size=4, language_balance=False, start_by_longest=True)
    loader = model.get_data_loader(config, is_eval=False, samples=samples, verbose=False)
    lengths = [s["audio_length"] for s in loader.dataset.samples]
    assert lengths[0] == max(lengths)                                  # 첫 배치에 가장 긴 발화
    assert _batches(loader) == _batches(loader)                        # 메모리 재기용 고정 순서


def test_language_balance_uses_coqui_weighted_sampler(tmp_path: Path) -> None:
    samples, chars = _sampler_samples(tmp_path)
    model, config = _ps_vits(tmp_path, chars, batch_size=4)
    loader = model.get_data_loader(config, is_eval=False, samples=samples, verbose=False)
    sampler = loader.batch_sampler
    assert isinstance(sampler, train_yourtts.BucketBatchSampler)
    assert type(sampler.sampler).__name__ == "WeightedRandomSampler" and sampler.batch_size == 4


@pytest.mark.parametrize("balance", [True, False])
def test_evaluation_reads_every_row_once_with_eval_batch_size(tmp_path: Path, balance: bool) -> None:
    """검증은 언어 균형과 상관없이 표본 추출기 없이 한 번씩 차례로 읽고 --eval-batch-size 를 쓴다."""
    samples, chars = _sampler_samples(tmp_path)
    model, config = _ps_vits(tmp_path, chars, batch_size=16, eval_batch_size=3, language_balance=balance)
    loader = model.get_data_loader(config, is_eval=True, samples=samples, verbose=False)
    batches = _batches(loader)
    assert [len(b) for b in batches] == [3] * 6 + [2]                  # 20줄, 마지막 배치도 씀
    assert sorted(i for b in batches for i in b) == list(range(20)) and batches == _batches(loader)
    assert not model._ps_loading_eval


def test_train_needs_g2p_settings_from_phonemize(tmp_path: Path) -> None:
    man = _manifest_dir(tmp_path)
    phonemize_manifest.main(["--manifest-dir", str(man)])
    rows = [r for s in make_manifest.SPLITS for r in make_manifest.read_manifest(make_manifest.manifest_path(man, s))]
    compute_dvectors.compute(rows, FakeEncoderManager(), man / "d_vectors.pth", base=man)
    (man / g2p.SETTINGS_NAME).unlink()
    args = train_yourtts.parse_args(["--manifest-dir", str(man), "--out", str(tmp_path / "runs"), "--device", "cpu",
                                     "--arch", "tiny"])
    with pytest.raises(FileNotFoundError, match="g2p.json"):
        train_yourtts.train(args)
    assert not (tmp_path / "runs").exists()                            # 학습 폴더를 만들기 전에 멈춘다


def test_init_from_checkpoint_skips_mismatched_layers(tmp_path: Path) -> None:
    src = Vits(_config(tmp_path, arch="tiny"))
    path = tmp_path / "src.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = Vits(train_yourtts.build_config(characters=list("abcdefg"), d_vector_file=str(_dvec_file(tmp_path)),
                                          output_path=str(tmp_path), arch="tiny"))
    info = train_yourtts.init_from_checkpoint(dst, path)
    assert info["shape_mismatch"] == ["text_encoder.emb.weight"] and info["kept_init"] == ["text_encoder.emb.weight"]
    assert info["loaded"] == len(src.state_dict()) - 1 and info["unexpected"] == 0
    torch.testing.assert_close(dst.duration_predictor.pre.weight, src.duration_predictor.pre.weight)
    assert not torch.equal(dst.text_encoder.emb.weight[:3], src.text_encoder.emb.weight[:3])


def test_init_from_checkpoint_reads_old_weight_norm_names(tmp_path: Path) -> None:
    """공개 YourTTS 처럼 옛 weight_norm 이름(weight_g, weight_v)과 화자 인코더 층이 있는 체크포인트."""
    src = Vits(_config(tmp_path, arch="tiny"))
    old: dict[str, torch.Tensor] = {}
    for k, v in src.state_dict().items():
        if k.endswith("parametrizations.weight.original0"):
            k = k.replace("parametrizations.weight.original0", "weight_g")
        elif k.endswith("parametrizations.weight.original1"):
            k = k.replace("parametrizations.weight.original1", "weight_v")
        old[k] = v
    assert any(k.endswith("weight_v") for k in old)
    old["speaker_encoder.conv1.weight"] = torch.zeros(3)
    path = tmp_path / "old.pth"
    torch.save({"model": old}, path)
    torch.manual_seed(7)
    dst = Vits(_config(tmp_path, arch="tiny"))
    info = train_yourtts.init_from_checkpoint(dst, path)
    assert info["kept_init"] == [] and info["shape_mismatch"] == [] and info["unexpected"] == 1
    torch.testing.assert_close(dst.waveform_decoder.ups[0].weight, src.waveform_decoder.ups[0].weight)
    torch.testing.assert_close(dst.posterior_encoder.enc.in_layers[0].weight,
                               src.posterior_encoder.enc.in_layers[0].weight)


def test_train_cli_defaults_are_paper_values() -> None:
    args = train_yourtts.parse_args(["--manifest-dir", "x"])
    assert args.device == "cuda" and args.epochs == 200 and args.batch_size == 48 and args.lr == 2e-4
    assert args.arch == "yourtts" and args.init_from is None and args.max_steps is None
    with pytest.raises(SystemExit):
        train_yourtts.parse_args(["--manifest-dir", "x", "--batch-size", "0"])
