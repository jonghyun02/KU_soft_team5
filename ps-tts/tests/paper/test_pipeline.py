"""논문 판 pipeline: 진짜 approx 프런트엔드(g2pK, phonemizer) + 가짜 LaBSE, COMET, 모델 1, 모델 2 로 끝까지 돌린다.

다루는 것: ISO 세 갈래(초벌 통과, 바꿔 쓰기 첫 통과에서 멈춤, 60번 상한 뒤 LaBSE 대안), 바꿔 쓰기 바탕은 늘 T0,
PS 후보 모으기(모자라면 더 묻기, ISO 문장 자체는 넣지 않음), 프레임 0개 후보 빼기, PS(식 11)와 PS-Comet(식 20),
결과 파일과 재생(FileBackend)이 같은 결정, ISO 오류 줄, 모델 2 멈춤, 설정 오류, 모델 닫기, 쓰지 않는 칸 알림,
Your-TTS 글자 목록 거르기(VocabFilter).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pstts.llm import FileBackend
from pstts.records import LineItem
from pstts.paper import g2p
from pstts.paper.config import PaperConfig
from pstts.paper.frontend import ApproxFrontend
from pstts.paper.io import result_dict, write_outputs
from pstts.paper.pipeline import (STATUS_ERROR, STATUS_OK, PaperLine, VocabFilter, check_config, open_frontend,
                                  run_paper, unused_columns)

EN_A, EN_B, EN_C = "We have to leave before the storm hits.", "I didn't order two tickets.", "Run!"
T0_A = "폭풍이 오기 전에 떠나야 해."              # 11음절, K 119.6 (K' 125.2): 초벌 통과
T0_B = "나는 티켓 두 장을 주문하지 않았어요."      # K 163.1 (K' 97.4): 길이 탈락
T0_C = "도망쳐!"                                  # K 32.6 (K' 13.9): 길이 통과, LaBSE 0.6 으로 뜻 탈락
ISO_B = {1: ["나는 표 두 장을 절대로 주문하지 않았어요.",    # 길이 탈락
             "표 두 장 주문 안 했어."],                    # 길이 통과, 뜻 탈락
         2: ["표 두 장은 주문 안 했어요.",                 # 둘 다 통과(3번째 검사)
             "표는 주문 안 했어."]}                        # 앞에서 멈춰 검사하지 않음
ISO_C = {1: ["어서 도망쳐!", "뛰어!"], 2: ["얼른 도망가!", "빨리 뛰어!"]}
LABSE = {T0_C: 0.6, "표 두 장 주문 안 했어.": 0.5, "어서 도망쳐!": 0.74, "뛰어!": 0.70, "얼른 도망가!": 0.74,
         "빨리 뛰어!": 0.5}
PS = {("a", 1): ["폭풍 전에 가자.", "폭풍이 치기 전에 떠나자.", "비가 오기 전에 가야 해.", "PVC."],
      ("b", 1): ["표 두 장 주문 안 했어.", "표 두 장은 주문 안 했어요.", "티켓 두 장은 안 샀어요.", "두 장 주문한 적 없어요."],
      ("c", 1): ["도망쳐!", "뛰어!"],
      ("c", 2): ["도망쳐!", "빨리 도망가!", "어서 뛰어!"]}
CFG = PaperConfig(device="cpu", iso_batch=2, iso_max=4, ps_n=4, ps_extra_rounds=1, ps_chunk=2)


def _script(stage: str, rnd: int, req) -> list[str]:
    if stage == "iso":
        return {"b": ISO_B, "c": ISO_C}[req.id].get(rnd, [])
    return PS.get((req.id, rnd), [])


def _items() -> list[LineItem]:
    return [LineItem(id="a", source_en=EN_A, register="반말", relationship="친구"),
            LineItem(id="b", source_en=EN_B, draft_ko=T0_B, lip=False),
            LineItem(id="c", source_en=EN_C)]


def _run(fakes, cfg: PaperConfig = CFG, backend=None, logs: list[str] | None = None, **kw):
    drafter = fakes.Drafter({EN_A: T0_A, EN_C: T0_C})
    labse, comet = fakes.Labse(LABSE), fakes.Comet()
    gen = backend if backend is not None else fakes.Gen(_script)
    made = {"drafter": drafter, "labse": labse, "comet": comet, "gen": gen}
    lines, info = run_paper(kw.pop("items", None) or _items(), cfg, backend=gen, make_drafter=lambda: drafter,
                            make_labse=lambda: labse, make_comet=lambda: comet,
                            log=(logs.append if logs is not None else (lambda m: None)), **kw)
    return lines, info, made


@pytest.fixture(scope="module")
def front() -> ApproxFrontend:
    return ApproxFrontend(CFG)


# ---------------------------------------------------------------- 끝까지

def test_end_to_end_iso_ps_and_ps_comet(fakes) -> None:
    logs: list[str] = []
    lines, info, made = _run(fakes, logs=logs)
    a, b, c = lines
    # 모델 1 은 초벌이 없는 줄만 한 번에 번역하고 닫힌다
    assert made["drafter"].seen == [[EN_A, EN_C]] and made["drafter"].closed
    assert made["labse"].closed and made["comet"].closed

    # ISO: a 는 초벌 통과, b 는 3번째 검사에서 통과하고 멈춤, c 는 4번(iso_max) 뒤 LaBSE 대안(같은 값이면 먼저 나온 것)
    assert (a.iso.status, a.iso.target, a.iso.iterations) == ("draft", T0_A, 0)
    assert (b.iso.status, b.iso.target, b.iso.iterations) == ("paraphrase", "표 두 장은 주문 안 했어요.", 3)
    assert [x.checked for x in b.iso.candidates] == [True, True, True, True, False]
    assert b.iso.candidates[-1].iteration is None
    assert (c.iso.status, c.iso.target, c.iso.iterations) == ("fallback", "어서 도망쳐!", 4)
    assert "LaBSE 가 가장 높은 후보" in c.message
    assert a.iso.k_prime_basis == "tts" and abs(a.iso.r - 1.0) < 1e-12
    assert b.iso.length_ratio == 0.6 and c.iso.length_ratio == 0.43

    # 바꿔 쓰기 요청: 바탕은 늘 T0, avoid 는 T0 와 앞 후보 전부, 줄마다 min(iso_batch, iso_max - 검사 수)
    gen = made["gen"]
    iso_reqs = [r for r in gen.requests if r[0] == "iso"]
    assert [(r[1], r[2], r[3]) for r in iso_reqs] == [(1, "b", 2), (1, "c", 2), (2, "b", 2), (2, "c", 2)]
    assert iso_reqs[0][4] == [T0_B] and iso_reqs[2][4] == [T0_B, *ISO_B[1]]
    iso_user = gen.prompts[0][1]
    assert '"ko": "' + T0_B + '"' in iso_user and '"length_ratio": 0.6' in iso_user
    assert '"en": "' + EN_B + '"' in iso_user

    # PS: 1차에 줄마다 ps_n 개, 모자란 c 만 2차에 모자란 만큼(avoid = 받은 후보). ISO 문장은 avoid 에 넣지 않는다
    ps_reqs = [r for r in gen.requests if r[0] == "ps"]
    assert [(r[1], r[2], r[3], r[4]) for r in ps_reqs] == [
        (1, "a", 4, []), (1, "b", 4, []), (1, "c", 4, []), (2, "c", 2, PS[("c", 1)])]
    ps_users = [u for _, u in gen.prompts if '"length_ratio"' not in u]
    assert ps_users and all('"en"' not in u for u in ps_users)
    assert [x.text for x in c.ps_candidates] == ["도망쳐!", "뛰어!", "빨리 도망가!", "어서 뛰어!"]
    assert c.ps_rounds == [1, 1, 2, 2]
    assert "표 두 장은 주문 안 했어요." in [x.text for x in b.ps_candidates]     # 모델이 낸 ISO 문장은 후보다

    # 프레임 0개 후보(g2pK 가 한글로 못 바꾼 PVC. 는 0길이)는 DTW None 으로 빼고, 나머지에서 PS = DTW 최소, PS-Comet = 식 20 최대
    assert a.ps_candidates[3].n_frames == 0 and a.ps_candidates[3].dtw is None
    assert a.choice.excluded == [3] and "뺀 후보 1개" in a.message
    for p in lines:
        valid = [i for i, x in enumerate(p.ps_candidates) if x.dtw is not None]
        dtws = np.array([p.ps_candidates[i].dtw for i in valid])
        assert p.choice.ps == valid[int(np.argmin(dtws))]
        lo, hi = dtws.min(), dtws.max()
        for i in valid:
            x = p.ps_candidates[i]
            inv = 1.0 if hi == lo else 1 - (x.dtw - lo) / (hi - lo)
            assert x.dtw_inv == pytest.approx(inv) and x.score == pytest.approx(1.6 * inv + 0.4 * x.comet)
        best = max(valid, key=lambda i: (p.ps_candidates[i].score, -i))
        assert p.choice.ps_comet == best and p.final_ko == p.ps_candidates[best].text
        assert p.status == STATUS_OK and p.n_src_frames > 0

    # COMET 은 한 번에 모든 PS 후보(원문, 후보)를 잰다
    (srcs, mts), = made["comet"].calls
    assert len(mts) == 4 + 4 + 4 and srcs[:4] == [EN_A] * 4 and srcs[-1] == EN_C

    # 기록
    assert info.llm == {"iso": {"calls": 2, "cache_hits": 0}, "ps": {"calls": 2, "cache_hits": 0}}
    assert info.iso_counts == {"draft": 1, "paraphrase": 1, "fallback": 1, "error": 0}
    assert info.status_counts == {STATUS_OK: 3}
    assert {"model1", "frontend", "iso_tts", "iso_labse", "iso_llm", "ps_llm", "ps_dtw", "comet", "total"} <= set(
        info.seconds)
    assert info.frontend["name"] == "approx" and info.device == "cpu"
    assert info.unused_columns == {"register": 1, "lip": 1, "relationship": 1}
    assert any("논문 판은 이 칸을 쓰지 않습니다: register 1줄, lip 1줄, relationship 1줄" in m for m in logs)
    assert {r["stage"] for r in info.rows} == {"iso", "ps"}


def test_outputs_and_replay_give_same_decisions(fakes, tmp_path: Path) -> None:
    lines, info, _ = _run(fakes)
    paths = write_outputs(tmp_path / "first", lines, info, CFG, ["id", "source_en", "draft_ko", "register"], "시험")
    assert all(p.is_file() for p in paths.values())
    rows = [json.loads(x) for x in paths["candidates.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert rows == info.rows and all(set(r) == {"id", "stage", "round", "candidates"} for r in rows)

    replay = FileBackend(paths["candidates.jsonl"])
    lines2, info2, _ = _run(fakes, backend=replay)
    assert [result_dict(p) for p in lines2] == [result_dict(p) for p in lines]
    assert info2.rows == info.rows and info2.llm_calls == 0
    write_outputs(tmp_path / "again", lines2, info2, CFG, ["id", "source_en", "draft_ko", "register"], "시험")
    for name in ("results.jsonl", "results.csv", "candidates.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "again" / name).read_bytes()


def test_main_version_candidate_file_gives_no_paper_candidates(fakes, tmp_path: Path) -> None:
    """stage 칸이 없는(기존 판) 후보 파일은 논문 판 단계에서 쓰지 않는다: 바꿔 쓰기 대안, PS 실패."""
    f = tmp_path / "main.jsonl"
    f.write_text(json.dumps({"id": "b", "round": 1, "candidates": ["표 두 장은 주문 안 했어요."]}, ensure_ascii=False)
                 + "\n", encoding="utf-8")
    lines, info, _ = _run(fakes, backend=FileBackend(f))
    assert [p.iso.status for p in lines] == ["draft", "fallback", "fallback"]
    assert all(p.status == STATUS_ERROR and not p.ps_candidates for p in lines)
    assert all("PS 후보를 하나도 받지 못해" in p.message for p in lines)
    assert any("iso 1차 후보가 없음" in e for e in info.llm_errors)


# ---------------------------------------------------------------- 실패와 멈춤

def test_iso_error_line_gets_no_ps(fakes) -> None:
    items = [LineItem(id="a", source_en=EN_A), LineItem(id="x", source_en="Hello there.")]
    lines, info, made = _run(fakes, items=items)          # 'Hello there.' 의 초벌은 가짜 모델 1 이 빈 글로 줌
    a, x = lines
    assert x.iso.status == "error" and x.iso.target is None and x.status == STATUS_ERROR
    assert "초벌(T0)이 비어" in x.message and x.final is None and not x.ps_candidates
    assert [r[2] for r in made["gen"].requests if r[0] == "ps"] == ["a"]
    assert a.status == STATUS_OK and info.status_counts == {STATUS_OK: 1, STATUS_ERROR: 1}


@pytest.mark.parametrize("stage", ["iso", "ps"])
def test_llm_unavailable_stops_and_records(fakes, stage: str) -> None:
    lines, info, made = _run(fakes, backend=fakes.Gen(_script, stop_stage=stage))
    assert "모델 2 를 쓸 수 없음" in info.llm_stopped
    assert all(p.status == STATUS_ERROR and p.final is None for p in lines)
    if stage == "iso":
        assert [p.iso.status for p in lines] == ["draft", "fallback", "fallback"]
        assert not any(r[0] == "ps" for r in made["gen"].requests)       # ISO 에서 멈추면 PS 는 묻지 않는다
    else:
        assert [p.iso.status for p in lines] == ["draft", "paraphrase", "fallback"]
    assert made["comet"].calls == [] and made["labse"].closed


def test_all_zero_frame_candidates_cannot_be_chosen(fakes) -> None:
    def script(stage, rnd, req):
        return ["PVC.", "XYZ!"] if stage == "ps" else _script(stage, rnd, req)
    items = [LineItem(id="a", source_en=EN_A)]
    lines, _, _ = _run(fakes, cfg=replace(CFG, ps_n=2), backend=fakes.Gen(script), items=items)
    (a,) = lines
    assert a.choice.ps is None and a.choice.ps_comet is None and a.choice.excluded == [0, 1]
    assert a.status == STATUS_ERROR and "프레임이 0개라 고르지 못함" in a.message


@pytest.mark.parametrize("change, words", [
    ({"tts": "yourtts"}, "--tts-dir"), ({"tts": "wav"}, "tts 는"), ({"llm": "gpt"}, "llm 은"),
    ({"sakoe_chiba_radius": -1}, "sakoe_chiba_radius"), ({"ps_n": 0}, "ps_n"), ({"iso_batch": 0}, "iso_batch"),
    ({"alpha": -0.1}, "alpha"), ({"labse_min": 1.5}, "labse_min")])
def test_bad_config_stops_before_models(fakes, change: dict, words: str) -> None:
    made: list[str] = []
    with pytest.raises(ValueError, match=words):
        run_paper(_items(), replace(CFG, **change), backend=fakes.Gen(_script),
                  make_drafter=lambda: made.append("drafter"), make_labse=lambda: made.append("labse"),
                  make_comet=lambda: made.append("comet"), make_front=lambda: made.append("front"), log=lambda m: None)
    assert made == []
    with pytest.raises(ValueError):
        check_config(replace(CFG, **change))


def test_frontend_preflight_stops_before_model_1(fakes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """진짜 프런트엔드를 쓸 때 그 준비물(모음 대응, yourtts 폴더)이 틀리면 모델 1, LaBSE, 모델 2 를 부르기 전에 멈춘다."""
    import pstts.paper.frontend as frontend

    drafter, gen = fakes.Drafter(default=T0_A), fakes.Gen(_script)
    kw = dict(backend=gen, make_drafter=lambda: drafter, make_labse=fakes.Labse, make_comet=fakes.Comet,
              log=lambda m: None)
    with pytest.raises(FileNotFoundError, match="config.json"):
        run_paper(_items(), replace(CFG, tts="yourtts", tts_dir=str(tmp_path)), **kw)    # 빈 체크포인트 폴더

    def broken() -> dict[str, str]:
        raise RuntimeError("모음 대응을 만들지 못함")

    monkeypatch.setattr(frontend, "en_label_map", broken)
    with pytest.raises(RuntimeError, match="모음 대응"):
        run_paper(_items(), CFG, **kw)
    assert drafter.seen == [] and gen.calls == 0 and gen.requests == []


def test_bad_items_stop_before_models(fakes) -> None:
    made: list[str] = []
    kw = dict(backend=fakes.Gen(_script), make_drafter=lambda: made.append("d"), make_labse=lambda: made.append("l"),
              make_comet=lambda: made.append("c"), make_front=lambda: made.append("f"), log=lambda m: None)
    for items in ([], [LineItem(id="a", source_en=EN_A), LineItem(id="a", source_en=EN_B)],
                  [LineItem(id="a", source_en=EN_A, seconds=0.0)]):
        with pytest.raises(ValueError):
            run_paper(items, CFG, **kw)
    assert made == []


class _Recorder:
    """프런트엔드를 감싸 닫힘을 적는다."""

    def __init__(self, inner) -> None:
        self.inner, self.name, self.frame_sec, self.closed = inner, inner.name, inner.frame_sec, False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def close(self) -> None:
        self.closed = True


def test_models_are_closed_when_a_stage_fails(fakes, front: ApproxFrontend) -> None:
    class BadLabse(fakes.Labse):
        def similarity(self, srcs, tgts):
            raise RuntimeError("가짜 LaBSE 고장")
    rec, labse = _Recorder(front), BadLabse()
    with pytest.raises(RuntimeError, match="가짜 LaBSE 고장"):
        run_paper(_items(), CFG, backend=fakes.Gen(_script), make_drafter=lambda: fakes.Drafter(default=T0_A),
                  make_labse=lambda: labse, make_comet=fakes.Comet, make_front=lambda: rec, log=lambda m: None)
    assert labse.closed and rec.closed


def test_seconds_column_sets_k_prime(fakes) -> None:
    items = [LineItem(id="a", source_en=EN_A, seconds=1.6)]
    lines, _, _ = _run(fakes, items=items)
    iso = lines[0].iso
    assert iso.k_prime_basis == "seconds" and iso.k_prime == pytest.approx(1.6 / CFG.frame_sec_sa)
    assert iso.r == pytest.approx(1.6 / iso.src_dur_sec)


def test_unused_columns_counts_only_real_values() -> None:
    items = [LineItem(id="a", source_en=EN_A), LineItem(id="b", source_en=EN_B, register="해요체", lip=False,
                                                         target_syllables=9, prev_en="Hi.", meaning_notes="부정")]
    assert unused_columns(items[:1]) == {}
    assert unused_columns(items) == {"register": 1, "lip": 1, "target_syllables": 1, "prev_en": 1,
                                     "meaning_notes": 1}


def test_unused_columns_count_values_kept_as_written() -> None:
    """논문 판 읽기가 기존 판 꼴에 맞지 않아 extra 에 그대로 둔 값도 '쓰지 않는 칸의 값' 이다."""
    from pstts.paper.pipeline import raw_unused

    item = LineItem(id="c", source_en=EN_A, extra={"register": "존댓말", "lip": "yes", "split": "dev"})
    assert unused_columns([item]) == {"register": 1, "lip": 1}
    assert raw_unused(item, "register") == "존댓말" and raw_unused(item, "split") == ""


# ---------------------------------------------------------------- Your-TTS 글자 목록 거르기

class _VocabFront:
    """Your-TTS 프런트엔드 꼴의 가짜: 글자 목록에 'Z' 와 'q' 가 없다."""
    name, frame_sec = "yourtts", 256 / 22050

    def __init__(self) -> None:
        self.closed = False
        self.seen: list[list[str]] = []

    def tokens(self, text: str, lang: str) -> list[str]:
        return g2p.with_blanks(list(text))

    def unknown_tokens(self, toks) -> list[str]:
        return list(dict.fromkeys(t for t in toks if t in ("Z", "q")))

    def durations(self, seqs, lang):
        self.seen.extend(seqs)
        return [np.ones(len(s)) for s in seqs]

    def labels(self, toks, lang):
        return [None] * len(toks)

    def cost(self, a, b):
        return np.zeros((len(a), len(b)))

    def info(self) -> dict:
        return {"name": self.name}

    def close(self) -> None:
        self.closed = True


def test_vocab_filter_drops_unknown_like_coqui_tokenizer() -> None:
    inner = _VocabFront()
    f = VocabFilter(inner)
    assert f.name == "yourtts" and f.frame_sec == inner.frame_sec
    toks = f.tokens("aZbq", "en")
    assert toks == g2p.with_blanks(["a", "b"])           # 버린 뒤 빈 토큰을 다시 끼움(처음, 사이, 끝)
    assert f.tokens("ab", "en") == g2p.with_blanks(["a", "b"])
    f.tokens("aZbq", "en")                               # 같은 글은 한 번만 센다
    f.tokens("Zq", "ko")
    assert f.tokens("ZZ", "ko") == [g2p.BLANK]
    info = f.info()
    assert info["dropped_texts"] == {"en": 1, "ko": 2}
    assert info["dropped_tokens"] == {"en": {"Z": 1, "q": 1}, "ko": {"Z": 3, "q": 1}}
    assert "TTSTokenizer" in info["unknown_token_rule"]
    (d,) = f.durations([toks], "en")
    assert d.shape == (len(toks),)
    f.close()
    assert inner.closed


def test_open_frontend_wraps_only_yourtts() -> None:
    front = open_frontend(PaperConfig(), "cpu")
    try:
        assert isinstance(front, ApproxFrontend)
    finally:
        front.close()


def test_paper_line_properties_without_choice() -> None:
    from pstts.paper.iso import IsoLine
    p = PaperLine(item=LineItem(id="z", source_en=EN_A), iso=IsoLine(id="z", source_en=EN_A, draft_ko=""))
    assert p.ps is None and p.final is None and p.final_ko == ""
