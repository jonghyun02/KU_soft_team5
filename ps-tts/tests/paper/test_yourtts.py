"""paper.yourtts: 작은 무작위 VITS(학습 레시피의 tiny 구조, CPU)로 토큰->id, 길이(Vits.inference 와 같은 경로, 결정성,
모양), m_p 모양, 모르는 글자 오류, 화자·언어, 화자 평균 d-vector 파일, 모음표 라벨과 비용, 모델을 올리기 전 G2P(g2p.json)와
모음표 확인, 체크포인트 찾기와 경로 고치기를 본다."""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pstts.config import ROOT
from pstts.paper import g2p
from pstts.paper.config import PaperConfig
from pstts.paper.frontend import Frontend, make_frontend
from pstts.paper.vowel_embed import build_vowel_table, save_vowel_table
from pstts.paper.yourtts import (SPEAKER_MEANS_NAME, UnknownTokenError, YourTTSFrontend, _check_config,
                                 find_checkpoint, fix_config_paths, write_speaker_means)

sys.path.insert(0, str(ROOT / "scripts" / "yourtts"))
import train_yourtts  # noqa: E402

from TTS.tts.configs.vits_config import VitsConfig  # noqa: E402
from TTS.tts.models.vits import Vits  # noqa: E402
from TTS.tts.utils.managers import save_file  # noqa: E402

TEXTS = {
    "en": ["We have to leave before the storm hits.", "I didn't order two tickets.", "Hello, world!"],
    "ko": ["폭풍이 오기 전에 떠나야 해.", "같이 읽고 신라에 가자.", "의사 회의 희망"],
}


def _vocab() -> list[str]:
    toks = {t for lang, texts in TEXTS.items() for s in texts for t in g2p.phonemes(s, lang)}
    return sorted(toks, key=ord)


@pytest.fixture(scope="module")
def tts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """학습 레시피 설정(train_yourtts.build_config, tiny)으로 만든 무작위 VITS 체크포인트 폴더.
    길이 예측기(SDP) 가중치를 시드 고정 정규분포로 흔들어 토큰마다 다른 길이가 나오게 한다.
    학습 폴더처럼 g2p.json(기본 G2P)과 이 체크포인트로 만든 모음표(vowel_table.json)를 둔다. 화자 평균 파일은 없다."""
    d = tmp_path_factory.mktemp("yourtts_tiny")
    rng = np.random.default_rng(0)
    dvec = {f"en#/clip_{i}.wav": {"name": "spk_a" if i < 2 else "spk_b",
                                  "embedding": rng.standard_normal(512).tolist()} for i in range(4)}
    save_file(dvec, str(d / "d_vectors.pth"))
    cfg = train_yourtts.build_config(characters=_vocab(), d_vector_file=str(d / "d_vectors.pth"), output_path=str(d),
                                     arch="tiny")
    torch.manual_seed(0)
    model = Vits(cfg)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.duration_predictor.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 0, "epoch": 0}, d / "best_model.pth")
    cfg.save_json(str(d / "config.json"))
    g2p.write_settings(d / g2p.SETTINGS_NAME)
    maker = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d)), "cpu", load_table=False)
    try:
        save_vowel_table(build_vowel_table(maker, TEXTS["ko"], TEXTS["en"], k=5, seed=0), d / "vowel_table.json")
    finally:
        maker.close()
    return d


def _copy(tts_dir: Path, tmp_path: Path) -> Path:
    d = tmp_path / "m"
    shutil.copytree(tts_dir, d)
    return d


@pytest.fixture(scope="module")
def front(tts_dir: Path) -> YourTTSFrontend:
    return YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir)), "cpu")


# ---------------------------------------------------------------------------- 만들기
def test_make_frontend_gives_yourtts(tts_dir: Path) -> None:
    f = make_frontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir)), "cpu")
    try:
        assert isinstance(f, YourTTSFrontend) and isinstance(f, Frontend)
        assert f.name == "yourtts" and f.device == "cpu"
        assert f.frame_sec == 256 / 22050
        assert Path(f.checkpoint_path).name == "best_model.pth"
    finally:
        f.close()
    assert f.model is None


def test_needs_tts_dir_and_valid_device(tts_dir: Path) -> None:
    with pytest.raises(ValueError):
        YourTTSFrontend(PaperConfig(tts="yourtts"), "cpu")
    with pytest.raises(ValueError):          # resolve_device 이름 검사(GPU 를 부르기 전에 멈춤)
        YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir)), "gpu")


# ---------------------------------------------------------------------------- 토큰 -> id
def test_ids_match_coqui_tokenizer(front: YourTTSFrontend) -> None:
    """BLANK -> blank id, 글자 -> 글자 id. 학습 때 Coqui TTSTokenizer.text_to_ids(phonemes) 와 같은 열."""
    tok = front.model.tokenizer
    for lang, texts in TEXTS.items():
        for text in texts:
            toks = front.tokens(text, lang)
            ids = front.ids(toks, lang)
            assert ids == tok.text_to_ids("".join(g2p.phonemes(text, lang)), lang)
            assert ids[0::2] == [front.blank_id] * (len(toks) // 2 + 1)
            assert [tok.characters.id_to_char(i) for i in ids[1::2]] == toks[1::2]


def test_unknown_characters_raise_korean_error(front: YourTTSFrontend) -> None:
    toks = front.tokens("Zebra quiz", "en")
    bad = front.unknown_tokens(toks)
    assert bad and all(t not in front._char_to_id for t in bad)
    assert bad == list(dict.fromkeys(bad))
    with pytest.raises(UnknownTokenError) as e:
        front.ids(toks, "en")
    assert e.value.tokens == bad and e.value.lang == "en"
    assert "모델 글자 목록에 없는 토큰" in str(e.value) and repr(bad[0]) in str(e.value)
    with pytest.raises(UnknownTokenError):
        front.durations([toks], "en")
    assert front.unknown_tokens(front.tokens(TEXTS["ko"][0], "ko")) == []


# ---------------------------------------------------------------------------- 길이
def test_durations_shape_determinism_and_seconds(front: YourTTSFrontend) -> None:
    seqs = [front.tokens(t, "en") for t in TEXTS["en"]]
    d1 = front.durations(seqs, "en")
    d2 = front.durations(seqs, "en")
    frames = front.duration_frames(seqs, "en")
    for s, a, b, f in zip(seqs, d1, d2, frames):
        assert a.shape == (len(s),) and a.dtype == np.float64
        assert np.array_equal(a, b)                                  # noise_scale_dp 0 이라 결정적
        assert np.all(np.isfinite(a)) and np.all(a >= 0)
        np.testing.assert_allclose(a, f * 256 / 22050, rtol=1e-12)
    assert len(set(np.round(frames[0], 4))) > 10                     # 토큰마다 다른 길이(빈 토큰도 길이를 받음)
    assert frames[0][0] > 0


def test_durations_equal_vits_inference(front: YourTTSFrontend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Vits.inference 가 올림(ceil)하기 직전의 w 와 같고, inference 의 durations 는 그 올림이다."""
    for lang in ("en", "ko"):
        toks = front.tokens(TEXTS[lang][0], lang)
        ours = front.duration_frames([toks], lang)[0]
        seen: list[torch.Tensor] = []
        real_ceil = torch.ceil

        def spy(x: torch.Tensor, *a, **k) -> torch.Tensor:
            seen.append(x.detach().clone())
            return real_ceil(x, *a, **k)

        monkeypatch.setattr(torch, "ceil", spy)
        ids = torch.tensor([front.ids(toks, lang)])
        out = front.model.inference(ids, aux_input={"x_lengths": torch.tensor([ids.shape[1]]),
                                                    "d_vectors": front._speaker["tensor"], "speaker_ids": None,
                                                    "language_ids": front._lang_ids[lang]})
        monkeypatch.setattr(torch, "ceil", real_ceil)
        w = seen[0][0, 0].double().numpy()
        np.testing.assert_array_equal(ours, w)
        np.testing.assert_array_equal(out["durations"][0, 0].double().numpy(), np.ceil(ours))


def test_language_changes_durations(front: YourTTSFrontend) -> None:
    toks = front.tokens(TEXTS["en"][0], "en")
    en = front.duration_frames([toks], "en")[0]
    as_ko = front._frames(toks, "ko")               # 같은 토큰에 한국어 언어 임베딩
    assert not np.allclose(en, as_ko)


def test_d_vector_is_l2_normalised_like_coqui(front: YourTTSFrontend) -> None:
    """BaseTTS._get_speaker_conditioning 은 d-vector 를 F.normalize 한다. 크기만 바꾸면 길이가 같다."""
    toks = front.tokens(TEXTS["ko"][1], "ko")
    base = front.duration_frames([toks], "ko")[0]
    saved = front._speaker["tensor"]
    try:
        front._speaker["tensor"] = saved * 3.0
        # float32 정규화 반올림이 확률 길이 예측기(흐름 + exp)를 지나며 1e-5 상대 오차 안팎으로 커진다
        np.testing.assert_allclose(front.duration_frames([toks], "ko")[0], base, rtol=1e-4, atol=1e-6)
        front._speaker["tensor"] = -saved
        assert not np.allclose(front.duration_frames([toks], "ko")[0], base)
    finally:
        front._speaker["tensor"] = saved
    aux = {"d_vectors": saved * 3.0, "speaker_ids": None}
    g = front.model._get_speaker_conditioning(aux, "emb_g", normalize_embedding=False)
    torch.testing.assert_close(g, F.normalize(saved).unsqueeze(-1))


def test_noise_and_length_scale_follow_config(tts_dir: Path, front: YourTTSFrontend) -> None:
    toks = front.tokens(TEXTS["en"][1], "en")
    base = front.duration_frames([toks], "en")[0]
    f2 = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir), length_scale=2.0), "cpu")
    try:
        np.testing.assert_allclose(f2.duration_frames([toks], "en")[0], 2.0 * base, rtol=1e-6)
    finally:
        f2.close()
    f3 = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir), noise_scale_dp=0.8), "cpu")
    try:
        torch.manual_seed(0)
        a = f3.duration_frames([toks], "en")[0]
        b = f3.duration_frames([toks], "en")[0]
        assert not np.array_equal(a, b)          # 잡음이 있으면 부를 때마다 다르다
        assert f3.info()["noise_scale_dp"] == pytest.approx(0.8)
    finally:
        f3.close()


def test_empty_text_has_one_blank(front: YourTTSFrontend) -> None:
    toks = front.tokens("", "ko")
    assert toks == [g2p.BLANK]
    (d,) = front.durations([toks], "ko")
    assert d.shape == (1,) and d[0] > 0
    with pytest.raises(ValueError):
        front.durations([[]], "ko")


# ---------------------------------------------------------------------------- m_p
def test_encoder_means_shape_and_value(front: YourTTSFrontend) -> None:
    toks = front.tokens(TEXTS["ko"][0], "ko")
    m = front.encoder_means(toks, "ko")
    assert m.shape == (16, len(toks))                       # tiny 구조 hidden_channels 16
    model = front.model
    with torch.no_grad():
        x = torch.tensor([front.ids(toks, "ko")])
        lang_emb = model.emb_l(front._lang_ids["ko"]).unsqueeze(-1)
        _, m_p, _, _ = model.text_encoder(x, torch.tensor([x.shape[1]]), lang_emb=lang_emb)
    np.testing.assert_allclose(m, m_p[0].double().numpy(), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------- 화자
def test_speaker_selection(tts_dir: Path) -> None:
    def pick(name: str | None) -> dict:
        f = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(tts_dir), tts_speaker=name), "cpu")
        try:
            return {k: v for k, v in f._speaker.items() if k != "tensor"} | {"vec": f._speaker["tensor"].numpy()}
        finally:
            f.close()

    first = pick(None)
    assert first["kind"] == "speaker_mean" and first["name"] == "spk_a" and first["n_clips"] == 2
    assert first["d_vector_file"] == "per_clip"
    by_name = pick("spk_b")
    assert by_name["name"] == "spk_b" and not np.allclose(by_name["vec"], first["vec"])
    clip = pick("en#/clip_3.wav")
    assert clip["kind"] == "clip" and clip["name"] == "spk_b" and clip["key"] == "en#/clip_3.wav"
    with pytest.raises(ValueError):
        pick("nobody")


def test_speaker_means_file_gives_the_same_vector(tts_dir: Path, tmp_path: Path) -> None:
    """화자 평균 파일(train_yourtts.py 가 학습 폴더에 씀)이 있으면 그것을 올린다. 벡터는 발화별 파일의 평균과 같다."""
    d = _copy(tts_dir, tmp_path)
    want = {}
    for name in (None, "spk_b"):
        f = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d), tts_speaker=name), "cpu")
        try:
            want[name] = f._speaker["tensor"].numpy().copy()
        finally:
            f.close()
    info = write_speaker_means(d / "d_vectors.pth", d / SPEAKER_MEANS_NAME)
    assert info["speakers"] == 2 and info["clips"] == 4
    for name in (None, "spk_b"):
        f = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d), tts_speaker=name), "cpu")
        try:
            spk = f._speaker
            np.testing.assert_array_equal(spk["tensor"].numpy(), want[name])
            assert spk["kind"] == "speaker_mean" and spk["n_clips"] == 2 and spk["d_vector_file"] == "speaker_means"
            assert len(f.model.speaker_manager.embeddings) == 2
            assert any("화자 평균" in k for k in f.info()["path_changes"])
        finally:
            f.close()
    clip = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d), tts_speaker="en#/clip_3.wav"), "cpu")
    try:                                                    # 발화 키를 고르면 발화별 파일을 쓴다
        assert clip._speaker["kind"] == "clip" and clip._speaker["d_vector_file"] == "per_clip"
        assert len(clip.model.speaker_manager.embeddings) == 4
    finally:
        clip.close()


# ---------------------------------------------------------------------------- G2P 와 모음표는 모델보다 먼저 확인
def test_g2p_file_is_checked_before_the_model_loads(tts_dir: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    d = _copy(tts_dir, tmp_path)
    loads: list[object] = []
    monkeypatch.setattr(Vits, "load_checkpoint", lambda *a, **k: loads.append(a))
    (d / g2p.SETTINGS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="g2p.json"):
        YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d)), "cpu")
    g2p.write_settings(d / g2p.SETTINGS_NAME, "en-us", False)
    with pytest.raises(ValueError, match="--g2p-en-voice en-us"):
        YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d)), "cpu")
    assert loads == []
    monkeypatch.undo()
    f = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d), g2p_en_voice="en-us"), "cpu")
    try:
        assert f.tokens("hello", "en") == g2p.tokens("hello", "en", en_voice="en-us")
        info = f.info()
        assert info["g2p"]["en"]["voice"] == "en-us" and info["g2p"]["trained_with"]["en"]["voice"] == "en-us"
    finally:
        f.close()


def test_check_folder_reads_g2p_and_table_without_the_model(tts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """파이프라인이 모델 1 앞에서 부르는 폴더 확인(pipeline.preflight_frontend)은 모델을 올리지 않는다."""
    from pstts.paper.yourtts import check_folder

    loads: list[object] = []
    monkeypatch.setattr(Vits, "load_checkpoint", lambda *a, **k: loads.append(a))
    folder = check_folder(PaperConfig(tts="yourtts", tts_dir=str(tts_dir)))
    assert folder.table is not None and folder.checkpoint.name == "best_model.pth" and len(folder.checkpoint_sha1) == 40
    assert folder.en_voice == "en-gb" and folder.en_stress is False and folder.g2p_settings["scheme"] == g2p.SCHEME
    assert check_folder(PaperConfig(tts="yourtts", tts_dir=str(tts_dir)), load_table=False).table is None
    assert loads == []


def test_tokens_follow_default_g2p(front: YourTTSFrontend) -> None:
    assert front.tokens("Hello, world!", "en") == g2p.tokens("Hello, world!", "en")      # en-gb, 강세표 없음
    assert front.tokens("안녕", "ko") == g2p.with_blanks(["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"])
    assert front.info()["g2p"]["scheme"] == g2p.SCHEME


# ---------------------------------------------------------------------------- 모음표
def test_labels_and_cost_with_vowel_table(tts_dir: Path, tmp_path: Path) -> None:
    d = _copy(tts_dir, tmp_path)
    (d / "vowel_table.json").unlink()
    cfg = PaperConfig(tts="yourtts", tts_dir=str(d))
    with pytest.raises(FileNotFoundError, match="모음표"):                # 만들 때 멈춘다(LaBSE, 모델 2 를 부르기 전)
        YourTTSFrontend(cfg, "cpu")
    f = YourTTSFrontend(cfg, "cpu", load_table=False)                    # 표를 만드는 쪽
    try:
        with pytest.raises(FileNotFoundError):
            f.labels(f.tokens(TEXTS["en"][0], "en"), "en")
        doc = build_vowel_table(f, TEXTS["ko"], TEXTS["en"], k=5, seed=0)
        assert doc["checkpoint"]["sha1"] == f.checkpoint_sha1
        assert set(doc["ko"]) <= set(f.vocab_vowels("ko")) and set(doc["en"]) <= set(f.vocab_vowels("en"))
        assert doc["dim"] == 16
        save_vowel_table(doc, d / "vowel_table.json")
        en_toks = f.tokens(TEXTS["en"][0], "en")
        ko_toks = f.tokens(TEXTS["ko"][0], "ko")
        en_l, ko_l = f.labels(en_toks, "en"), f.labels(ko_toks, "ko")
        assert [t for t, x in zip(en_toks, en_l) if x is not None] == [t for t in en_toks if g2p.is_vowel(t, "en")]
        assert all(x == t for t, x in zip(ko_toks, ko_l) if x is not None)
        assert en_l[0] is None and ko_l[0] is None                       # 빈 토큰은 null
        c = f.cost(en_l, ko_l)
        assert c.shape == (len(en_toks), len(ko_toks))
        i_v = next(i for i, x in enumerate(en_l) if x is not None)
        j_v = next(j for j, x in enumerate(ko_l) if x is not None)
        assert c[0, 0] == 0.0
        assert c[0, j_v] == c[i_v, 0] == pytest.approx(doc["max_distance"])
        assert c[i_v, j_v] == pytest.approx(f.table.distance(ko_l[j_v], en_l[i_v]))
        # 표에 없는 모음은 null 로 보고 센다
        doc2 = dict(doc, en=[e for e in doc["en"] if e != "ə"],
                    matrix=[[v for e, v in zip(doc["en"], row) if e != "ə"] for row in doc["matrix"]])
        save_vowel_table(doc2, d / "vowel_table.json")
        f._table = None
        labs = f.labels(en_toks, "en")
        assert all(x is None for t, x in zip(en_toks, labs) if t == "ə")
        assert f.missing_vowels["en"]["ə"] == en_toks.count("ə") > 0
        info = f.info()
        json.dumps(info, ensure_ascii=False)
        assert info["vowel_table"]["en"] == doc2["en"] and info["missing_vowels"]["en"]["ə"] > 0
    finally:
        f.close()


def test_vowel_table_from_other_checkpoint_is_rejected(tts_dir: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    d = _copy(tts_dir, tmp_path)
    doc = {"ko": ["ㅏ"], "en": ["a"], "matrix": [[1.0]], "checkpoint": {"path": "x", "sha1": "0" * 40}}
    save_vowel_table(doc, d / "vowel_table.json")
    loads: list[object] = []
    monkeypatch.setattr(Vits, "load_checkpoint", lambda *a, **k: loads.append(a))
    with pytest.raises(ValueError, match="다른 체크포인트"):              # 모델을 올리기 전에 멈춘다
        YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d)), "cpu")
    assert loads == []
    monkeypatch.undo()
    f = YourTTSFrontend(PaperConfig(tts="yourtts", tts_dir=str(d)), "cpu", load_table=False)
    try:
        with pytest.raises(ValueError, match="다른 체크포인트"):
            f.cost(["a"], ["ㅏ"])
    finally:
        f.close()


# ---------------------------------------------------------------------------- 폴더와 설정
def test_find_checkpoint_order(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path)
    (tmp_path / "checkpoint_9.pth").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="checkpoint_9.pth"):
        find_checkpoint(tmp_path)
    (tmp_path / "model_file.pth").write_bytes(b"x")
    assert find_checkpoint(tmp_path).name == "model_file.pth"
    (tmp_path / "best_model_100.pth").write_bytes(b"x")
    (tmp_path / "best_model_2000.pth").write_bytes(b"x")
    assert find_checkpoint(tmp_path).name == "best_model_2000.pth"
    (tmp_path / "best_model.pth").write_bytes(b"x")
    assert find_checkpoint(tmp_path).name == "best_model.pth"
    assert find_checkpoint(tmp_path, "checkpoint_9.pth").name == "checkpoint_9.pth"
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path, "nope.pth")


def test_fix_config_paths_uses_tts_dir(tts_dir: Path, tmp_path: Path) -> None:
    config = VitsConfig()
    config.load_json(str(tts_dir / "config.json"))
    shutil.copyfile(tts_dir / "d_vectors.pth", tmp_path / "d_vectors.pth")
    config.model_args.d_vector_file = "/elsewhere/old_run/d_vectors.pth"
    config.d_vector_file = ["/elsewhere/old_run/d_vectors.pth"]
    config.model_args.speaker_encoder_model_path = "/elsewhere/model_se.pth"
    changes = fix_config_paths(config, tmp_path)
    want = str((tmp_path / "d_vectors.pth").resolve())
    assert config.model_args.d_vector_file == want and config.d_vector_file == [want]
    assert config.model_args.speaker_encoder_model_path == "" and not config.model_args.use_speaker_encoder_as_loss
    assert set(changes) == {"model_args.d_vector_file", "config.d_vector_file", "model_args.speaker_encoder"}
    config.model_args.d_vector_file = "/elsewhere/missing.pth"
    with pytest.raises(FileNotFoundError):
        fix_config_paths(config, tmp_path)


@pytest.mark.parametrize("field,value", [("use_phonemes", True), ("text_cleaner", "multilingual_cleaners"),
                                         ("add_blank", False), ("enable_eos_bos_chars", True)])
def test_check_config_rejects_other_recipes(tts_dir: Path, field: str, value: object) -> None:
    config = VitsConfig()
    config.load_json(str(tts_dir / "config.json"))
    _check_config(config)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="레시피"):
        _check_config(config)


def test_info_is_json(front: YourTTSFrontend) -> None:
    info = front.info()
    text = json.dumps(info, ensure_ascii=False)
    assert info["name"] == "yourtts" and info["noise_scale_dp"] == 0.0 and info["length_scale"] == 1.0
    assert info["languages"]["ids"] == {"en": 0, "ko": 1} and info["use_sdp"] is True
    assert info["checkpoint_sha1"] == front.checkpoint_sha1 and len(front.checkpoint_sha1) == 40
    assert "tensor" not in info["speaker"] and "tensor" not in text
    assert replace(PaperConfig(), tts="yourtts").noise_scale_dp == 0.0
