"""sinkprep.validate 시험: 예시 파일 통과 + 규칙마다 일부러 어긴 사본 거부 + catalog·needs_review."""
import copy
import csv
import hashlib
import json
import random
import shutil
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from sinkprep import paths, validate as V

KINDS = ("lines", "script", "asr", "align", "final")


# ---------------------------------------------------------------- 도구

class Ex:
    """examples/ 를 tmp 로 복사해 두고, 고친 사본을 만들어 검사한다."""

    def __init__(self, root: Path):
        self.root = root
        shutil.copytree(paths.EXAMPLES, root)
        self.lines = root / "fm_example" / "lines.jsonl"
        self.script = root / "fm_example" / "script.json"
        self.asr = root / "asr.json"
        self.align = root / "align.json"
        self.final = root / "final.json"
        self.k = 0

    def load(self, p: Path):
        if p.suffix == ".jsonl":
            return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
        return json.loads(p.read_text(encoding="utf-8"))

    def edit(self, p: Path, fn) -> Path:
        """p 를 읽어 fn(data) 로 고친 사본 경로(같은 폴더, 이름만 다름).

        fn 은 data 를 제자리에서 고친다. JSONL(행 목록)만 fn 이 돌려준 새 목록으로 바꿀 수 있다.
        """
        d = self.load(p)
        r = fn(d)
        if isinstance(d, list) and isinstance(r, list):
            d = r
        self.k += 1
        out = p.with_name(f"mut{self.k}_{p.name}")
        if p.suffix == ".jsonl":
            out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in d), encoding="utf-8")
        else:
            out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return out

    # 검사 함수(짝 파일 포함)
    def v_lines(self, p=None):
        return V.validate_lines(p or self.lines)

    def v_script(self, p=None, lines=None):
        return V.validate_script(p or self.script, lines or self.lines)

    def v_asr(self, p=None):
        return V.validate_asr(p or self.asr)

    def v_final(self, p=None, script="default"):
        return V.validate_final(p or self.final, self.script if script == "default" else script)

    def v_align(self, p=None, final=None, asr=None):
        return V.validate_align(p or self.align, final or self.final, asr or self.asr, self.script)


@pytest.fixture
def ex(tmp_path):
    return Ex(tmp_path / "ex")


def has(errs, code):
    assert any(e.startswith(f"[{code}]") for e in errs), f"[{code}] 오류가 없음: {errs}"


def item(d, iid):
    return next(it for it in d["items"] if it["id"] == iid)


def el(d, eid):
    return next(e for e in d["elements"] if e["id"] == eid)


# ---------------------------------------------------------------- 스키마·예시

@pytest.mark.parametrize("kind", KINDS)
def test_schema_files_are_valid_2020_12(kind):
    from jsonschema import Draft202012Validator

    s = V.load_schema(kind)
    assert s["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(s)


def test_examples_pass(ex):
    assert ex.v_lines() == []
    assert ex.v_script() == []
    assert ex.v_asr() == []
    assert ex.v_final() == []
    assert ex.v_align() == []


def test_examples_cover_spec_cases(ex):
    """예시 파일이 명세의 주요 경우를 실제로 담고 있는지(빈 예시로 통과하는 것 방지)."""
    s = ex.load(ex.script)
    types = {e["type"] for e in s["elements"]}
    assert types == {"front", "scene", "action", "dialogue", "onscreen", "transition"}
    f = ex.load(ex.final)
    sts = {it.get("status") for it in f["items"] if it["type"] == "dialogue"}
    assert sts == {"same", "changed", "added", "absent"}
    assert any(it.get("ko_src") == "mt" for it in f["items"])
    # 시각 없는 항목이 시각 있는 항목 사이에 끼어 있음(D7 허용)
    timed = [it["start"] is not None for it in f["items"]]
    k = timed.index(False)
    assert any(timed[:k]) and any(timed[k + 1:])
    assert "…" not in ex.final.read_text(encoding="utf-8") + ex.asr.read_text(encoding="utf-8") \
        + ex.align.read_text(encoding="utf-8")


def test_run_on_examples_returns_0(capsys):
    rc = V.run(SimpleNamespace(paths=[str(paths.EXAMPLES)], docs=None, limit=None, force=False, workers=1))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "실패 0개" in out


def test_run_reports_failures_and_returns_1(ex, capsys):
    bad = ex.edit(ex.final, lambda d: d.pop("schema"))
    rc = V.run(SimpleNamespace(paths=[str(bad)], docs=None, limit=None, force=False, workers=1))
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "[E1]" in out


# ---------------------------------------------------------------- E1·E2

@pytest.mark.parametrize("name", ["script", "asr", "align", "final"])
def test_E1_schema_field_required(ex, name):
    p = getattr(ex, name)
    bad = ex.edit(p, lambda d: d.pop("schema"))
    errs = getattr(ex, f"v_{name}")(bad)
    has(errs, "E1")


def test_E1_wrong_schema_value(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: d.update(schema="sink.final/9"))), "E1")
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d.update(schema="sink.final/1"))), "E1")


def test_E1_lines_header_required(ex):
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows[1:])), "E1")
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows[0].pop("schema"))), "E1")


def test_E1_unknown_json_in_run(ex, capsys):
    p = ex.root / "other.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    kind, errs = V.check_file(p)
    has(errs, "E1")


def test_E2_nfd_text_rejected(ex):
    nfd = unicodedata.normalize("NFD", "거짓말하지 마.")
    bad = ex.edit(ex.script, lambda d: el(d, "E0008").update(text=nfd))
    has(ex.v_script(bad), "E2")

    def f(rows):
        rows[5]["text"] = unicodedata.normalize("NFD", rows[5]["text"])
    has(ex.v_lines(ex.edit(ex.lines, f)), "E2")


def test_E2_bom_and_non_utf8_rejected(ex):
    p = ex.root / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + ex.final.read_bytes())
    has(V.validate_final(p), "E2")
    q = ex.root / "cp949.json"
    q.write_bytes(ex.final.read_text(encoding="utf-8").encode("cp949", errors="replace"))
    has(V.validate_final(q), "E2")


# ---------------------------------------------------------------- lines

def test_lines_n_must_be_consecutive(ex):
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows.pop(4))), "B2")


def test_lines_file_index_in_range(ex):
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows[3].update(file=1))), "ref")


@pytest.mark.parametrize("change", [
    {"x": 1.2}, {"y": -0.1}, {"src": "pdf"}, {"conf": 0.5}, {"page": 0}, {"n": "3"},
    {"cell": {"table": 1, "row": 2}}, {"extra": 1}, {"bold": "yes"},
])
def test_lines_row_schema(ex, change):
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows[3].update(change))), "schema")


def test_lines_row_missing_field(ex):
    has(ex.v_lines(ex.edit(ex.lines, lambda rows: rows[3].pop("page"))), "schema")


def test_lines_ocr_row_with_conf_ok_and_cell_ok(ex):
    def f(rows):
        rows[3].update(src="ocr", conf=0.91)
        rows[4].update(cell={"table": 0, "row": 2, "col": 1})
    assert ex.v_lines(ex.edit(ex.lines, f)) == []


LINE_VALS = [None, True, False, -1, 0, 1, 2, 0.5, 1.5, -0.2, "", "x", "ocr", "txt", [], {},
             {"table": 0, "row": 0, "col": 0}, {"table": -1, "row": 0, "col": 0}, 3.0]


def test_line_fast_path_agrees_with_schema(ex):
    """빠른 줄 검사(_line_fast_ok)가 참이면 JSON Schema 도 반드시 통과해야 한다."""
    v = V.schema_validator("lines", "line")
    rows = ex.load(ex.lines)[1:]
    for r in rows:
        assert V._line_fast_ok(r) and v.is_valid(r)
    rng = random.Random(7)
    base = rows[3]
    n_fast = 0
    for r in _mutants(base, rng, 4000, LINE_VALS):
        if V._line_fast_ok(r):
            n_fast += 1
            assert v.is_valid(r), r
    assert n_fast > 50     # 빠른 길이 실제로 쓰임
    ocr = dict(base, src="ocr", conf=0.7)
    assert V._line_fast_ok(ocr) and v.is_valid(ocr)


def _mutants(base, rng, k, vals):
    keys = list(base) + ["extra"]
    out = []
    for _ in range(k):
        r = copy.deepcopy(base)
        for _ in range(rng.randint(1, 3)):
            key = rng.choice(keys)
            if rng.random() < 0.15 and key in r:
                del r[key]
            else:
                r[key] = copy.deepcopy(rng.choice(vals))
        out.append(r)
    return out


ELEMENT_VALS = [None, True, 0, -1, 2, 0.5, "", "x", "E0001", "E01", "SC001", "SC1", "P01", "P1", "ko", "KO",
                "en-US", "mixed", "on", "shout", "insert", "caption", "scene", "dialogue", "action", "note",
                [], [0], [0, 0], [-1], [1.5], ["P01"], ["P01", "P01"], ["x"], [{"text": "a", "at": 0}],
                [{"text": "a"}], [{"text": "a", "at": -1}], {}, {"text": "a", "at": 0}]


@pytest.mark.parametrize("eid", ["E0001", "E0004", "E0005", "E0007", "E0010", "E0013"])
def test_element_fast_path_agrees_with_schema(ex, eid):
    """빠른 요소 검사(_element_fast_ok)가 참이면 JSON Schema $defs/element 도 반드시 통과해야 한다."""
    v = V.schema_validator("script", "element")
    s = ex.load(ex.script)
    for e in s["elements"]:
        assert V._element_fast_ok(e) and v.is_valid(e), e
    base = el(s, eid)
    n_fast = 0
    for m in _mutants(base, random.Random(eid), 3000, ELEMENT_VALS):
        if V._element_fast_ok(m):
            n_fast += 1
            assert v.is_valid(m), m
    assert n_fast > 20


WORD_VALS = [None, True, -1, 0, 3, 0.5, 1.5, 723.4, "", "어제", "ko", "EN", "en", [], {}]


@pytest.mark.parametrize("kind", ["asr", "final"])
def test_word_fast_path_agrees_with_schema(ex, kind):
    v = V.schema_validator(kind, "word")
    fast = V._asr_word_fast_ok if kind == "asr" else V._final_word_fast_ok
    if kind == "asr":
        words = [w for p in ex.load(ex.asr)["passes"] for w in p["words"]]
    else:
        words = [w for it in ex.load(ex.final)["items"] for w in it.get("words", [])]
    for w in words:
        assert fast(w) and v.is_valid(w), w
    n_fast = 0
    for base in (words[0], words[-1]):
        for m in _mutants(base, random.Random(kind), 3000, WORD_VALS):
            if fast(m):
                n_fast += 1
                assert v.is_valid(m), m
    assert n_fast > 20


ITEM_VALS = [None, True, False, 0, -1, 1, 0.5, 1.5, 723.4, "", "x", "I0001", "I1", "SC001", "E0006", "E1",
             "P01", "ko", "en", "same", "changed", "added", "absent", "text", "translation", "mt", "human",
             "aligned", "estimated", "none", "script", "unknown", "on", "insert", "dialogue", "scene", "front",
             [], ["P01"], ["P01", "P01"], [{"w": "가", "s": 1.0, "e": 1.2, "p": 0.9}], [{"w": "", "s": 1, "e": 2, "p": 1}],
             [{"kind": "word", "script": "a", "heard": "b"}], [{"kind": "typo", "script": "a", "heard": "b"}],
             {"el": "E0006", "text": "t"}, {"el": "X", "text": "t"}, {}]


@pytest.mark.parametrize("iid", ["I0001", "I0003", "I0004", "I0005", "I0006", "I0008", "I0010"])
def test_item_fast_path_agrees_with_schema(ex, iid):
    """빠른 final 항목 검사(_item_fast_ok)가 참이면 JSON Schema $defs/item 도 반드시 통과해야 한다."""
    v = V.schema_validator("final", "item")
    f = ex.load(ex.final)
    for it in f["items"]:
        assert V._item_fast_ok(it) and v.is_valid(it), it
    n_fast = 0
    for m in _mutants(item(f, iid), random.Random(iid), 4000, ITEM_VALS):
        if V._item_fast_ok(m):
            n_fast += 1
            assert v.is_valid(m), m
    assert n_fast > 20


OP_VALS = [None, 0, -1, 2, 0.5, "", "I0003", "I3", "E0006", "same", "changed", "added", "absent", "text",
           "translation", "rule", "human", "plain:0", ["plain:0", "plain:3"], ["plain:0"], ["0", "3"], [],
           [{"text": "a", "src": "asr", "score": -1.0}], [{"text": "a", "src": "", "score": 1}], {}]


@pytest.mark.parametrize("k", range(4))
def test_op_fast_path_agrees_with_schema(ex, k):
    v = V.schema_validator("align", "op")
    a = ex.load(ex.align)
    for op in a["ops"]:
        assert V._op_fast_ok(op) and v.is_valid(op), op
    n_fast = 0
    for m in _mutants(a["ops"][k], random.Random(k), 4000, OP_VALS):
        if V._op_fast_ok(m):
            n_fast += 1
            assert v.is_valid(m), m
    assert n_fast > 20


def test_word_errors_found_beyond_first_word(ex):
    errs = ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003")["words"][2].update(p=1.5)))
    assert any(e.startswith("[schema] items/2/words/2/p") for e in errs), errs
    errs = ex.v_asr(ex.edit(ex.asr, lambda d: d["passes"][0]["words"][5].update(x=1)))
    assert any(e.startswith("[schema] passes/0/words/5") for e in errs), errs
    errs = ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0007").update(how="shout")))
    assert any(e.startswith("[schema] elements/6/how") for e in errs), errs


# ---------------------------------------------------------------- script (B2·B3·참조)

def test_B2_line_not_covered(ex):
    has(ex.v_script(ex.edit(ex.script, lambda d: d["dropped"].pop(0))), "B2")


def test_B2_line_covered_twice(ex):
    has(ex.v_script(ex.edit(ex.script, lambda d: d["dropped"].append({"lines": [6], "why": "page_header"}))), "B2")


def test_B2_line_out_of_range(ex):
    has(ex.v_script(ex.edit(ex.script, lambda d: d["dropped"].append({"lines": [99], "why": "blank"}))), "B2")


def test_B2_without_lines_still_checks_duplicates(ex):
    bad = ex.edit(ex.script, lambda d: el(d, "E0005").update(lines=[4]))
    has(V.validate_script(bad), "B2")
    assert V.validate_script(ex.script) == []


def test_B3_unknown_element_type(ex):
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0005").update(type="note"))), "schema")


@pytest.mark.parametrize("change", [
    {"how": "shout"}, {"speakers": ["형사"]}, {"lang": "Korean"}, {"sung": "no"},
    {"paren": [{"text": "웃으며"}]}, {"kind": "insert"},
])
def test_script_dialogue_fields(ex, change):
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0007").update(change))), "schema")


def test_script_element_type_specific_fields(ex):
    # action 에 대사 칸, onscreen 의 kind 값, scene 요소의 scene=null
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0005").update(speakers=["P01"]))), "schema")
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0010").update(kind="caption"))), "schema")
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0004").update(scene=None))), "schema")
    has(ex.v_script(ex.edit(ex.script, lambda d: d["scenes"][1].update(flags=["memory"]))), "schema")


def test_script_refs(ex):
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0006").update(speakers=["P09"]))), "ref")
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0005").update(scene="SC099"))), "ref")
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0006").update(id="E0005"))), "ref")
    has(ex.v_script(ex.edit(ex.script, lambda d: el(d, "E0007")["paren"][0].update(at=99))), "ref")


# ---------------------------------------------------------------- final (D2~D8)

def test_D2_start_before_end(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(end=725.6))), "D2")


def test_D2_words_nonempty(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(words=[]))), "D2")


def test_D2_words_inside_line(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004")["words"][1].update(e=727.0))), "D2")


def test_D2_heard_equals_words(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(heard="집에 없었는데요.",
                                                                        ko="집에 없었는데요."))), "D2")


def test_D3_absent_has_no_time_and_has_ref(ex):
    def timed(d):
        item(d, "I0006").update(start=727.0, end=727.5)
    has(ex.v_final(ex.edit(ex.final, timed)), "D3")
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0006").update(script_ref=None))), "D3")


def test_D3_added_has_no_script_ref(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0005").update(
        script_ref={"el": "E0008", "text": "거짓말하지 마."}))), "D3")


def test_D4_changed_needs_diffs(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003").update(diffs=[]))), "D4")


def test_D4_same_has_no_diffs(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(
        diffs=[{"kind": "word", "script": "집에", "heard": "집에서"}]))), "D4")


def test_D5_timed_dialogue_needs_ko(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0010").update(ko=None))), "D5")


def test_D5_korean_line_ko_equals_heard(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(ko="집에 있었어요."))), "D5")
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(ko_src="mt"))), "D5")


def test_D5_foreign_word_needs_mt(ex):
    def f(d):
        it = item(d, "I0004")
        it["words"][0]["lang"] = "en"
    has(ex.v_final(ex.edit(ex.final, f)), "D5")
    # 외국어 낱말이 섞인 한국어 줄은 ko_src=mt 이면 통과
    def g(d):
        it = item(d, "I0004")
        it["words"][0]["lang"] = "en"
        it["ko_src"] = "mt"
    assert ex.v_final(ex.edit(ex.final, g)) == []


def test_D6_translation_match_has_no_diffs(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003").update(match="translation"))), "D6")


def test_D7_timed_items_in_order(ex):
    def swap(d):
        a, b = item(d, "I0003"), item(d, "I0004")
        ia, ib = d["items"].index(a), d["items"].index(b)
        d["items"][ia], d["items"][ib] = b, a
        a["id"], b["id"] = "I0004", "I0003"
    has(ex.v_final(ex.edit(ex.final, swap)), "D7")


def test_D7_item_ids_follow_order(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(id="I0002"))), "ref")
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(id="I0001"))), "D7")


def test_D7_untimed_item_follows_script_order(ex):
    """시각 없는 항목(absent)은 대본 순서로 앞뒤 항목 사이에 놓인다(명세 D7). 예시의 I0006(E0008)은 E0007 과 E0009
    사이라 통과, 마지막 씬 앞(E0013 과 E0014 사이)으로 옮기면 D7."""
    assert [e for e in ex.v_final(ex.final) if e.startswith("[D7]")] == []

    def move(d):
        a = item(d, "I0006")
        d["items"].remove(a)
        d["items"].insert(len(d["items"]) - 1, a)
        for k, it in enumerate(d["items"], 1):
            it["id"] = f"I{k:04d}"
    has(ex.v_final(ex.edit(ex.final, move)), "D7")


def test_D8_refs_inside_file(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003").update(scene="SC009"))), "D8")
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003").update(speakers=["P07"]))), "D8")


def test_D8_refs_into_script(ex):
    def f(d):
        it = item(d, "I0004")
        it["script_ref"]["el"] = "E0999"
        it["src_el"] = "E0999"
    has(ex.v_final(ex.edit(ex.final, f)), "D8")
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(src_el="E0999"))), "D8")
    # 종류가 다른 요소를 가리킴(action 항목 → dialogue 요소)
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(src_el="E0006"))), "D8")
    # script.json 을 주지 않으면 대본 참조는 보지 않는다
    assert ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(src_el="E0999")), script=None) == []


def test_D8_src_el_matches_script_ref(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(src_el="E0006"))), "D8")


def test_final_onscreen_kind_only_on_onscreen(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(kind="insert"))), "schema")


def test_final_front_items_not_allowed(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(type="front"))), "schema")


def test_final_quality_counts(ex):
    has(ex.v_final(ex.edit(ex.final, lambda d: d["quality"].update(same=9))), "Q")


def test_final_time_three_decimals(ex):
    def f(d):
        it = item(d, "I0004")
        it["start"] = 725.6001
        it["words"][0]["s"] = 725.6001
    has(ex.v_final(ex.edit(ex.final, f)), "T")


def test_D4_same_and_changed_agree_with_the_text_key(ex):
    """글자로 맞춘 줄은 same 이면 heard 와 대본 글의 열쇠(공백·문장부호를 뺀 글자열)가 같고, changed 면 다르다."""
    # 통과: 띄어쓰기·문장부호만 다른 same (대본 글을 고친 사본이라 script.json 과 견주는 D9 는 빼고 본다)
    assert ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004")["script_ref"].update(text="집에있었는데요?!")),
                      script=None) == []
    # 실패: same 인데 글자가 다름
    errs = ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004")["script_ref"].update(text="집에 없었는데요.")))
    has(errs, "D4")
    assert any("status=same" in e and "다름" in e for e in errs)
    # 실패: changed 인데 띄어쓰기·문장부호만 다름
    errs = ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0003")["script_ref"].update(text="어제밤에, 어디 있었냐고?")))
    has(errs, "D4")
    assert any("status=changed" in e for e in errs)

    # 뜻으로 맞춘 줄(match=translation)은 글자를 견주지 않는다
    def translated(d):
        it = item(d, "I0010")
        it["match"] = "translation"
        it["script_ref"]["text"] = "선생님!"
    assert ex.v_final(ex.edit(ex.final, translated), script=None) == []


def test_D8_every_script_element_appears_exactly_once(ex):
    """대본의 scene·action·dialogue·onscreen·transition 요소는 빠짐없이 정확히 한 번씩 items 에 나온다
    (front·unknown 은 넣지 않는다: 예시의 front 요소 3개가 없어도 통과)."""
    def drop(d):
        d["items"].remove(item(d, "I0002"))
    errs = ex.v_final(ex.edit(ex.final, drop))
    has(errs, "D8")
    assert any("items 에 없는 대본 요소 1개" in e and "E0005" in e for e in errs)
    assert ex.v_final(ex.edit(ex.final, drop), script=None) == []      # script.json 을 못 찾으면 빠짐은 보지 않는다

    def twice(d):
        a = copy.deepcopy(item(d, "I0002"))
        a["id"] = f"I{len(d['items']) + 1:04d}"
        a["start"] = d["items"][-1]["start"]
        d["items"].append(a)
    errs = ex.v_final(ex.edit(ex.final, twice))
    has(errs, "D8")
    assert any("둘 이상" in e and "E0005" in e for e in errs)


def _last_time(d):
    return max(t for it in d["items"] for t in [it["start"], it["end"]] + [x for w in it.get("words", [])
                                                                              for x in (w["s"], w["e"])] if t is not None)


def test_D2_times_stay_inside_the_video(ex):
    """모든 시각은 film.video.duration 을 넘지 않는다(허용 오차 0.5초)."""
    assert ex.v_final(ex.edit(ex.final, lambda d: d["film"]["video"].update(
        duration=round(max(_last_time(d), d["scenes"][-1]["end"]) - 0.4, 3)))) == []
    errs = ex.v_final(ex.edit(ex.final, lambda d: d["film"]["video"].update(duration=round(_last_time(d) - 0.6, 3))))
    has(errs, "D2")
    assert any("영상 길이" in e for e in errs)
    # 씬 구간도 본다
    errs = ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][-1].update(end=d["film"]["video"]["duration"] + 1.0)))
    assert any(e.startswith("[D2]") and "scenes/" in e for e in errs)


def test_D2_word_starts_do_not_go_back(ex):
    """한 줄 안에서 낱말 시작 시각은 줄어들지 않는다."""
    def swap(d):
        ws = item(d, "I0004")["words"]
        ws[0]["s"], ws[1]["s"] = ws[1]["s"], ws[0]["s"]
        ws[0]["e"] = max(ws[0]["e"], ws[0]["s"])
    errs = ex.v_final(ex.edit(ex.final, swap))
    assert any(e.startswith("[D2]") and "앞 낱말보다 앞선" in e for e in errs), errs

    def same_start(d):                                                 # 같은 시작 시각은 괜찮다
        ws = item(d, "I0004")["words"]
        ws[1]["s"] = ws[0]["s"]
    assert ex.v_final(ex.edit(ex.final, same_start)) == []


def test_C2_film_id_equals_folder_name(ex, tmp_path, monkeypatch):
    """films/{film_id}/final.json 자리에 있는 파일은 film.id 가 폴더 이름과 같다. 일치율은 0~1(스키마)."""
    films = tmp_path / "films"
    monkeypatch.setattr(paths, "FILMS", films)
    for name in ("kmdb_K12345", "local_0001"):
        (films / name).mkdir(parents=True)
        shutil.copy(ex.final, films / name / "final.json")
    assert V.validate_final(films / "kmdb_K12345" / "final.json", ex.script) == []
    errs = V.validate_final(films / "local_0001" / "final.json", ex.script)
    has(errs, "C2")
    assert any("폴더 이름" in e for e in errs)
    assert ex.v_final() == []                                          # films 밖의 파일은 폴더 이름을 보지 않는다
    has(ex.v_final(ex.edit(ex.final, lambda d: d["film"]["script"].update(match=1.5))), "schema")
    assert ex.v_final(ex.edit(ex.final, lambda d: d["film"]["script"].update(match=None))) == []


# ---------------------------------------------------------------- asr (C1)

def test_C1_word_index_unique(ex):
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d["passes"][0]["words"][3].update(i=2))), "C1")


def test_C1_word_index_contiguous(ex):
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d["passes"][0]["words"].pop(5))), "C1")


def test_C1_pass_ids_unique(ex):
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d["passes"][1].update(id="plain"))), "C1")


def test_C3_asr_has_no_alignment_fields(ex):
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d.update(ops=[]))), "schema")
    has(ex.v_asr(ex.edit(ex.asr, lambda d: d["passes"][0]["words"][0].update(item="I0003"))), "schema")


def test_C1_align_refs_exist_in_asr(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(asr=["plain:0", "plain:99"]))), "C1")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(asr=["ctx:0", "plain:3"]))), "C1")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(asr=["plain:3", "plain:0"]))), "C1")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(asr=["0", "3"]))), "schema")


def test_C1_align_asr_sha1(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d.update(asr_sha1="0" * 40))), "C1")


def test_align_asr_and_final_same_video(ex):
    other = ex.edit(ex.final, lambda d: d["film"]["video"].update(sha1="b" * 40))
    errs = ex.v_align(final=other)
    assert any("video_sha1" in e for e in errs), errs


# ---------------------------------------------------------------- align (C2)

def test_C2_every_dialogue_item_has_one_op(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"].pop(1))), "C2")


def test_C2_no_duplicate_or_unknown_ops(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"].append(dict(d["ops"][1])))), "C2")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][1].update(item="I0999"))), "C2")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][1].update(item="I0002"))), "C2")


def test_C2_op_equals_status(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(op="same"))), "C2")


def test_C2_script_el_matches_final(ex):
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][1].update(script_el="E0008"))), "C2")


def test_align_op_shape_rules(ex):
    # added 인데 대본 요소가 있음 / absent 인데 낱말 범위가 있음
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][2].update(script_el="E0008"))), "schema")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][3].update(asr=["plain:0", "plain:1"]))), "schema")
    has(ex.v_align(ex.edit(ex.align, lambda d: d["ops"][0].update(decided_by="human"))), "schema")


def _c2(errs, *need):
    """[C2] 오류 가운데 need 글자열을 모두 담은 것이 있는가."""
    return any(e.startswith("[C2]") and all(n in e for n in need) for e in errs)


def _resigned(ex, asr_path):
    """asr.json 사본에 맞게 asr_sha1 을 다시 적은 align.json 사본(받아쓰기가 바뀐 뒤 align 만 다시 돌린 상태)."""
    sha = hashlib.sha1(asr_path.read_bytes()).hexdigest()
    return ex.edit(ex.align, lambda d: d.update(asr_sha1=sha))


def test_C2_final_words_are_the_words_of_the_asr_range(ex):
    """대사 항목의 words(w, s, e, p)는 그 op 의 asr 범위 낱말과 같다(명세 C2). 범위나 낱말이 어긋난 사본을 잡는다."""
    assert ex.v_align() == []                                           # 낱말의 lang 칸(I0010)은 견주지 않는다
    # asr 범위를 한 낱말 옆으로 옮김(범위는 여전히 asr.json 안이라 C1 은 통과)
    errs = ex.v_align(ex.edit(ex.align, lambda d: d["ops"][4].update(asr=["plain:11", "plain:15"])))
    assert _c2(errs, "I0007", "words", "plain:11~plain:15") and not any(e.startswith("[C1]") for e in errs)
    # final 쪽 낱말의 시각, 글자, 확률이 바뀜
    for change in ({"e": 725.85}, {"w": "집에서"}, {"p": 0.5}):
        errs = ex.v_align(final=ex.edit(ex.final, lambda d: item(d, "I0004")["words"][0].update(change)))
        assert _c2(errs, "I0004", "words", "0번째"), errs
    # 낱말 하나가 빠짐(heard 도 함께 고쳐 final 자체 검사는 통과하는 사본)
    def drop(d):
        it = item(d, "I0005")
        it["words"].pop()
        it["heard"] = it["ko"] = " ".join(w["w"] for w in it["words"])
        it["end"] = it["words"][-1]["e"]
    short = ex.edit(ex.final, drop)
    assert ex.v_final(short) == [] and _c2(ex.v_align(final=short), "I0005", "final 3개, asr 4개")
    # 글은 그대로이고 확률만 바뀐 받아쓰기: align 만 다시 돌린 뒤의 낡은 final.json
    asr2 = ex.edit(ex.asr, lambda d: d["passes"][0]["words"][4].update(p=0.5))
    errs = ex.v_align(_resigned(ex, asr2), asr=asr2)
    assert len(errs) == 1 and _c2(errs, "I0004", "words")
    assert "집에" not in errs[0]                                          # 오류 글에 대사를 적지 않는다
    # final 이나 asr 가운데 하나만 주면 견주지 않는다
    assert V.validate_align(ex.edit(ex.align, lambda d: d["ops"][4].update(asr=["plain:11", "plain:15"])),
                            None, ex.asr, ex.script) == []


def test_C2_word_times_are_compared_after_the_same_repair_as_final(ex):
    """timeline 이 바로잡는 시각(줄 안에서 거꾸로 간 시작, 영상 길이 밖)은 바로잡은 값으로 견준다."""
    # 낱말 plain:5 가 앞 낱말(725.6)보다 먼저 시작한다고 적힌 받아쓰기
    asr2 = ex.edit(ex.asr, lambda d: d["passes"][0]["words"][5].update(s=725.5))
    al2 = _resigned(ex, asr2)
    assert _c2(ex.v_align(al2, asr=asr2), "I0004", "words")              # 예전 값(725.9)을 가진 final
    raw = ex.edit(ex.final, lambda d: item(d, "I0004")["words"][1].update(s=725.5))
    assert _c2(ex.v_align(al2, final=raw, asr=asr2), "I0004", "words")   # 받아쓰기 값을 그대로 옮긴 final
    has(ex.v_final(raw), "D2")
    fixed = ex.edit(ex.final, lambda d: item(d, "I0004")["words"][1].update(s=725.6))
    assert ex.v_align(al2, final=fixed, asr=asr2) == [] and ex.v_final(fixed) == []
    # 영상 길이 밖의 낱말: 영상 끝으로 당긴 값이어야 한다
    def short_video(d):
        d["film"]["video"]["duration"] = 741.0
        d["scenes"] = [dict(s, start=min(s["start"], 741.0), end=min(s["end"], 741.0)) for s in d["scenes"]]
        for it in d["items"]:
            if it["start"] is not None and it["start"] > 741.0:
                it["start"] = 741.0
    late = ex.edit(ex.final, short_video)
    assert _c2(ex.v_align(final=late), "I0010", "words")

    def pulled(d):
        short_video(d)
        it = item(d, "I0010")
        it["words"][0].update(s=741.0, e=741.02)
        it.update(start=741.0, end=741.02)
    assert ex.v_align(final=ex.edit(ex.final, pulled)) == []


def test_C2_one_word_belongs_to_one_op(ex):
    """같은 pass 의 같은 낱말을 두 op 이 갖지 않는다(한 번 난 소리는 한 줄에만)."""
    # 낱말 plain:4 가 I0003 과 I0004 두 줄에
    twice = ex.edit(ex.align, lambda d: d["ops"][0].update(asr=["plain:0", "plain:4"]))
    errs = ex.v_align(twice)
    assert _c2(errs, "I0003", "I0004", "같은 낱말", "plain:4") and not any("집에" in e for e in errs)
    assert _c2(V.validate_align(twice, None, None, ex.script), "같은 낱말")         # align.json 만으로 알 수 있다
    # 다른 pass 의 낱말은 번호가 같아도 다른 낱말이다: I0003 을 ctx 의 0~3 으로 바꿔도(글·시각이 같은 낱말) 걸리지 않는다
    def to_ctx(d):
        d["ops"][0].update(asr=["ctx:0", "ctx:3"])
    def ctx_words(d):
        for w, p in zip(item(d, "I0003")["words"], (0.97, 0.94, 0.96, 0.9)):
            w["p"] = p
    assert ex.v_align(ex.edit(ex.align, to_ctx), final=ex.edit(ex.final, ctx_words)) == []
    # 서로 다른 pass 의 줄끼리 시간이 겹친 것은 시각만으로 잘못이라 단정할 수 없어 오류가 아니다(final 이 quality 에 남긴다).
    # 여기서는 added 줄(I0005)이 ctx 낱말을 가리키게 한 사본이라 낱말이 달라진 것만 걸린다
    errs = ex.v_align(ex.edit(ex.align, lambda d: d["ops"][2].update(asr=["ctx:0", "ctx:3"])))
    assert len(errs) == 1 and _c2(errs, "I0005", "words")


def test_C2_added_op_needs_words(ex):
    errs = ex.v_align(ex.edit(ex.align, lambda d: d["ops"][2].update(asr=None)))
    assert _c2(errs, "I0005", "op=added")


# ---------------------------------------------------------------- final ↔ script (D9)

def _d9(errs):
    return [e for e in errs if e.startswith("[D9]")]


def test_D9_fields_copied_from_the_script_match_it(ex):
    """대본 요소에서 복사한 칸(명세 4.7: paren, how, sung, overlap 과 화자·씬·대본 글)은 script.json 과 같다."""
    assert ex.v_final() == []
    for key, val in (("how", "phone"), ("sung", True), ("overlap", 1), ("paren", []), ("speakers", ["P01"]),
                     ("scene", "SC002")):
        errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update({key: val}))))
        assert len(errs) == 1 and errs[0].endswith(f"대본과 다름: {key}") and "I0004" in errs[0] and "E0007" in errs[0], errs
        assert "집에" not in errs[0] and "웃으며" not in errs[0]            # 칸 이름만 알린다
    # 여러 칸이 다르면 한 줄에 모아 알린다
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0009").update(how="phone", sung=False, paren=[]))))
    assert len(errs) == 1 and errs[0].endswith("how, paren, sung")
    # 영상에 없는 줄(absent)의 대본 글
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0006")["script_ref"].update(text="거짓말 마."))))
    assert len(errs) == 1 and errs[0].endswith("script_ref.text")
    # 대사 아닌 항목: 글, 씬, 화면 글자 종류
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0002").update(text="형사가 웃는다."))))[0].endswith(": text")
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0008").update(kind="title"))))[0].endswith(": kind")
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0011").update(scene="SC002"))))[0].endswith(": scene")
    # lang 은 글자로 맞춘 줄만 대본 그대로(뜻으로 맞춘 줄은 들린 말의 언어라 견주지 않는다)
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0010").update(lang="fr"))))[0].endswith(": lang")
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0010").update(lang="fr", match="translation")))) == []
    # 대본에 적힌 화자가 아닌데(앞 대사에서 이어받은 추정 화자) speaker_from 이 script
    guessed = ex.edit(ex.script, lambda d: el(d, "E0007").update(speaker_inferred=True))
    errs = _d9(ex.v_final(script=guessed))
    assert len(errs) == 1 and errs[0].endswith(": speaker_from") and "I0004" in errs[0]
    # script.json 을 못 찾으면 견주지 않는다. added 줄(대본 요소 없음)도 견줄 것이 없다
    assert ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0004").update(how="phone")), script=None) == []
    assert _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0005").update(how="phone")))) == []


def test_D9_scenes_and_people_copied_from_the_script_match_it(ex):
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][0].update(heading="취조실", flags=["dream"]))))
    assert len(errs) == 1 and "scenes/0(SC001)" in errs[0] and errs[0].endswith("heading, flags")
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["people"][1].update(name="최창혁", aliases=["창혁"]))))
    assert len(errs) == 1 and "people/1(P02)" in errs[0] and errs[0].endswith("name, aliases")
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["people"].append({"id": "P09", "name": "행인", "aliases": []}))))
    assert len(errs) == 1 and "people/4(P09)" in errs[0] and "script.json" in errs[0]
    assert ex.v_final(ex.edit(ex.final, lambda d: d["people"][1].update(name="최창혁")), script=None) == []


def test_D9_scene_span_agrees_with_the_items_of_the_scene(ex):
    """씬 구간의 start 는 그 씬 항목들의 시각 범위 안이다. 항목 시각과 무관한 구간을 잡는다."""
    assert ex.v_final(script=None) == []                    # 예시: 씬 머리의 시각에서 다음 씬 시작까지로 잡은 구간도 통과
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][0].update(start=1.0, end=2.0)), script=None))
    assert len(errs) == 1 and "scenes/0(SC001)" in errs[0] and "시각 범위" in errs[0]
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][0].update(start=760.0, end=790.2)), script=None))
    assert len(errs) == 1 and "시각 범위" in errs[0]          # 그 씬의 마지막 항목(742.0)보다 늦게 시작
    assert ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][0].update(start=723.4, end=741.6)), script=None) == []
    # 시각 있는 항목이 있는 씬인데 구간이 없음 / 없는 씬인데 구간이 있음
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: d["scenes"][1].update(start=None, end=None, timing="none")),
                          script=None))
    assert len(errs) == 1 and "scenes/1(SC002)" in errs[0] and "구간이 없음" in errs[0]
    errs = _d9(ex.v_final(ex.edit(ex.final, lambda d: item(d, "I0012").update(scene="SC001")), script=None))
    assert len(errs) == 1 and "scenes/1(SC002)" in errs[0] and "시각 있는 항목이 없는 씬" in errs[0]


# ---------------------------------------------------------------- catalog.csv / needs_review.csv

def _row(**kw):
    base = {c: "" for c in V.CATALOG_COLUMNS}
    base.update(category="ko_film", script_lang="ko", is_translation="false", version_label="unknown",
                is_primary="true", status="pending", real_format="hwp5")
    base.update(kw)
    return base


def _write_catalog(p: Path, rows, bom=True, columns=V.CATALOG_COLUMNS):
    with open(p, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """작은 가짜 원본·extracted 와 맞는 catalog 행들."""
    root = tmp_path / "proj"
    raw = root / "filmmakers_scripts"
    (raw / "한국영화 시나리오").mkdir(parents=True)
    (raw / "외국영화 시나리오").mkdir(parents=True)
    (root / "extracted" / "3").mkdir(parents=True)
    files = {
        "filmmakers_scripts/한국영화 시나리오/가 [1].hwp": b"a",
        "filmmakers_scripts/외국영화 시나리오/B [2].pdf": b"b",
        "filmmakers_scripts/한국영화 시나리오/묶음 [3].zip": b"z",
        "extracted/3/하나.hwp": b"c1",
        "extracted/3/둘.hwp": b"c2",
    }
    for rel, data in files.items():
        (root / rel).write_bytes(data)
    (raw / "manifest.csv").write_text("post_id,title\n", encoding="utf-8")
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "RAW", raw)
    monkeypatch.setattr(paths, "EXTRACTED", root / "extracted")
    monkeypatch.setattr(paths, "CATALOG", root / "catalog.csv")
    monkeypatch.setattr(paths, "QA", root / "qa")
    monkeypatch.setattr(paths, "NEEDS_REVIEW", root / "qa" / "needs_review.csv")
    monkeypatch.setattr(paths, "DOCS", root / "docs")
    monkeypatch.setattr(paths, "FILMS", root / "films")
    monkeypatch.setattr(paths, "EXAMPLES", root / "examples")

    def sha(rel):
        return hashlib.sha1(files[rel]).hexdigest()

    k = list(files)
    rows = [
        _row(doc_id="fm1", post_id="1", files=k[0], sha1=sha(k[0]), title_raw="가", film_id="local_0001"),
        _row(doc_id="fm2", post_id="2", category="foreign_film", files=k[1], sha1=sha(k[1]), real_format="pdf_text",
             title_raw="B", script_lang="en", film_id="tmdb_550", is_primary="true"),
        _row(doc_id="fm3", post_id="3", files=k[2], sha1=sha(k[2]), real_format="archive", status="excluded",
             is_primary="false", note="archive_container:extracted/3"),
        _row(doc_id="fm3_1", post_id="3", files=k[3], sha1=sha(k[3]), title_raw="하나", film_id="local_0003",
             dup_group="G001"),
        _row(doc_id="fm3_2", post_id="3", files=k[4], sha1=sha(k[4]), title_raw="둘", film_id="local_0003",
             dup_group="G001", is_primary="false", match="0.5", extract_quality="0.91", year="2004"),
    ]
    return SimpleNamespace(root=root, rows=rows, files=files, sha=sha)


def test_catalog_ok(fake_root):
    _write_catalog(paths.CATALOG, fake_root.rows)
    assert V.validate_catalog() == []
    assert V.validate_catalog(check_sha1=True) == []


def test_catalog_mixed_with_file_formats_ok(fake_root):
    rows = fake_root.rows
    rows[3].update(files=rows[3]["files"] + "|" + rows[4]["files"],
                   sha1=rows[3]["sha1"] + "|" + rows[4]["sha1"], real_format="mixed", file_formats="hwp5|hwp3")
    del rows[4]
    _write_catalog(paths.CATALOG, rows)
    assert V.validate_catalog() == []


def test_catalog_A1_missing_file(fake_root):
    _write_catalog(paths.CATALOG, fake_root.rows[:-1])
    errs = V.validate_catalog()
    has(errs, "A1")
    assert any("extracted/3/둘.hwp" in e for e in errs)
    assert V.validate_catalog(check_files=False) == []


def test_catalog_A1_file_in_two_rows(fake_root):
    rows = fake_root.rows
    rows[4]["files"] = rows[3]["files"]
    _write_catalog(paths.CATALOG, rows)
    has(V.validate_catalog(), "A1")


def test_catalog_A1_nonexistent_file(fake_root):
    rows = fake_root.rows
    rows[0]["files"] = "filmmakers_scripts/한국영화 시나리오/없음 [1].hwp"
    _write_catalog(paths.CATALOG, rows)
    errs = V.validate_catalog()
    assert any("없는 파일을 가리킴" in e for e in errs)


def test_catalog_manifest_not_required(fake_root):
    _write_catalog(paths.CATALOG, fake_root.rows)
    assert not any("manifest.csv" in e for e in V.validate_catalog())


def test_catalog_A2_one_primary_per_film(fake_root):
    rows = fake_root.rows
    rows[4]["is_primary"] = "true"
    _write_catalog(paths.CATALOG, rows)
    has(V.validate_catalog(), "A2")


@pytest.mark.parametrize("col,val", [
    ("real_format", "hwp"), ("category", "korean"), ("version_label", "v2"), ("status", "done!"),
    ("is_primary", "yes"), ("sha1", "abc"), ("film_id", "imdb_1"), ("match", "1.5"), ("doc_id", "fm9"),
    ("files", "C:\\x.hwp"), ("dup_group", "17"), ("year", "04"),
])
def test_catalog_bad_values(fake_root, col, val):
    rows = fake_root.rows
    rows[0][col] = val
    _write_catalog(paths.CATALOG, rows)
    has(V.validate_catalog(check_files=False), "catalog")


def test_catalog_files_sha1_count(fake_root):
    rows = fake_root.rows
    rows[0]["sha1"] = rows[0]["sha1"] + "|" + rows[1]["sha1"]
    _write_catalog(paths.CATALOG, rows)
    errs = V.validate_catalog(check_files=False)
    assert any("개수가 다름" in e for e in errs)


def test_catalog_header_and_bom(fake_root):
    cols = [c for c in V.CATALOG_COLUMNS if c != "file_formats"]
    _write_catalog(paths.CATALOG, fake_root.rows, columns=cols)
    assert any("헤더에 없는 칸" in e for e in V.validate_catalog(check_files=False))
    _write_catalog(paths.CATALOG, fake_root.rows, bom=False)
    assert any("BOM" in e for e in V.validate_catalog(check_files=False))


def _write_review(p: Path, rows, header=V.NEEDS_REVIEW_COLUMNS):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_needs_review_ok_and_bad(fake_root):
    ok = [["doc", "fm1", "", "", "extract_broken", "0.31", "글자층 깨짐", "docs/fm1/lines.jsonl"],
          ["film", "", "local_0003", "", "version_mismatch", "", "판본 확인", ""],
          ["item", "fm1", "local_0003", "I0004", "asr_low_conf", "0.4", "", "films/local_0003/final.json"]]
    _write_review(paths.NEEDS_REVIEW, ok)
    assert V.validate_needs_review() == []
    _write_review(paths.NEEDS_REVIEW, ok + [["page", "fm1", "", "", "broken", "x", "", ""]])
    errs = V.validate_needs_review()
    assert len([e for e in errs if e.startswith("[review]")]) >= 3   # level, problem, score
    _write_review(paths.NEEDS_REVIEW, ok + [["doc", "", "", "", "other", "", "", ""]])
    has(V.validate_needs_review(), "review")
    _write_review(paths.NEEDS_REVIEW, ok, header=("level", "doc_id", "problem"))
    has(V.validate_needs_review(), "review")


# ---------------------------------------------------------------- run(): 기본 대상 수집

def test_run_default_scope_and_B1_coverage(fake_root, capsys):
    _write_catalog(paths.CATALOG, fake_root.rows)
    src = Path(__file__).resolve().parent.parent / "examples" / "fm_example"
    lines = [json.loads(s) for s in (src / "lines.jsonl").read_text(encoding="utf-8").splitlines()]
    script = json.loads((src / "script.json").read_text(encoding="utf-8"))
    for did in ("fm1", "fm2", "fm3_1"):          # fm3_2 는 일부러 만들지 않음
        d = paths.DOCS / did
        d.mkdir(parents=True)
        lines[0]["doc_id"] = did
        script["doc_id"] = did
        (d / "lines.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                                       encoding="utf-8")
        (d / "script.json").write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    args = SimpleNamespace(paths=[], docs=None, limit=None, force=False, workers=1)
    rc = V.run(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "docs/fm3_2/lines.jsonl" in out and "[B1]" in out
    assert "lines: 3개 검사, 0개 실패" in out and "script: 3개 검사, 0개 실패" in out
    # --docs 로 고르면 B1 덮기·catalog 는 보지 않음
    rc = V.run(SimpleNamespace(paths=[], docs="fm1,fm2", limit=None, force=False, workers=1))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "lines: 2개 검사" in out
    # 폴더 이름과 doc_id 가 다르면 잡음
    (paths.DOCS / "fm1" / "script.json").write_text(json.dumps(dict(script, doc_id="fm2"), ensure_ascii=False),
                                                    encoding="utf-8")
    rc = V.run(SimpleNamespace(paths=[], docs="fm1", limit=None, force=False, workers=1))
    assert rc == 1
    assert "폴더 이름" in capsys.readouterr().out


def _film_row(fake_root, **kw):
    """fake_root 의 첫 문서(fm1 = local_0001)에 영상을 등록한 catalog 를 쓴다."""
    rows = fake_root.rows
    rows[0].update(video_path="videos/local_0001.mp4", video_sha1="aa7a5bf94b487a638410204d4878d7cbebbeb702",
                   video_duration="7290.400", video_lang="ko", match="0.710", status="done")
    rows[0].update(kw)
    _write_catalog(paths.CATALOG, rows)


def _put_final(film_id="local_0001", **script_kw):
    """예시 final.json 을 films/{film_id}/final.json 으로(film.id 와 대표 문서를 맞춰서)."""
    d = json.loads((Path(__file__).resolve().parent.parent / "examples" / "final.json").read_text(encoding="utf-8"))
    d["film"]["id"] = film_id
    d["film"]["script"].update(doc_id="fm1")
    d["film"]["script"].update(script_kw)
    p = paths.FILMS / film_id / "final.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def test_D1_film_with_video_needs_final_json(fake_root, capsys):
    """인자 없이 돌리면 영상이 있는 영화(대표 문서)마다 films/{film_id}/final.json 이 있어야 한다."""
    _write_catalog(paths.CATALOG, fake_root.rows)
    assert V._film_coverage_errors() == []                             # 영상이 없는 영화는 final 이 없어도 된다
    _film_row(fake_root)
    errs = V._film_coverage_errors()
    assert len(errs) == 1 and errs[0].startswith("[D1] local_0001: films/local_0001/final.json 이 없음")
    args = SimpleNamespace(paths=[], docs=None, limit=None, force=False, workers=1)
    assert V.run(args) == 1
    out = capsys.readouterr().out
    assert "films (D1 덮기)" in out and "[D1] local_0001" in out
    V.run(SimpleNamespace(paths=[], docs="fm1", limit=None, force=False, workers=1))
    assert "[D1]" not in capsys.readouterr().out                       # --docs 로 고르면 덮기는 보지 않는다
    p = _put_final()
    assert V._film_coverage_errors() == []
    V.run(args)
    out = capsys.readouterr().out
    assert "[D1]" not in out and "final: 1개 검사, 0개 실패" in out
    # 대표가 아닌 문서, 제외 문서의 영상 칸은 보지 않는다
    p.unlink()
    _film_row(fake_root, is_primary="false")
    assert V._film_coverage_errors() == []


def test_D1_final_json_matches_its_catalog_row(fake_root):
    """final.json 이 catalog 행과 같은 대본·같은 영상의 것인지(대표 문서, 영상 sha1, 영상 길이, 일치율)."""
    _film_row(fake_root)
    _put_final()
    assert V._film_coverage_errors() == []
    for kw, word in (({"match": "0.500"}, "match"), ({"match": ""}, "match"), ({"video_sha1": "b" * 40}, "video.sha1"),
                     ({"video_duration": "7000.000"}, "duration")):
        _film_row(fake_root, **kw)
        errs = V._film_coverage_errors()
        assert len(errs) == 1 and errs[0].startswith("[D1] local_0001") and word in errs[0], (kw, errs)
    _film_row(fake_root)
    _put_final(doc_id="fm2")
    assert any("대표 문서" in e for e in V._film_coverage_errors())
    _put_final(match=None)                                             # 일치율이 없는 final 은 견주지 않는다
    assert V._film_coverage_errors() == []
