"""sinkprep.timeline: 항목 차례, 추정 시각, id, 씬 구간."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import align_synth as S  # noqa: E402
from sinkprep import timeline, validate  # noqa: E402


def op(kind, el=None, a=None, b=None, pid="plain"):
    return {"op": kind, "script_el": el, "asr": None if a is None else [f"{pid}:{a}", f"{pid}:{b}"]}


def words(*spans):
    """[(시작, 끝)] -> plain pass. 낱말 글은 w0, w1, ..."""
    return S.asr_of(S.say([(f"w{i}", s, e) for i, (s, e) in enumerate(spans)]))


SCRIPT = S.tiny_script([
    ("1", "scene", "S#1 집"), ("1", "action", "철수가 들어온다."), ("1", "dialogue", "다녀왔습니다."),
    ("1", "dialogue", "어, 왔니?"), ("1", "action", "가방을 내려놓는다. 한참 말이 없다."), ("1", "dialogue", "밥은?"),
    ("2", "scene", "S#2 방"), ("2", "onscreen", "달력"), ("2", "dialogue", "..."), ("2", "transition", "F.O"),
])


def test_build_timeline_orders_by_time_and_numbers_ids():
    """시각 있는 항목은 start 순, absent 는 대본 순서로 앞 항목 뒤, id 는 I0001 부터 차례대로."""
    asr = words((10.0, 10.5), (10.5, 11.0), (12.0, 12.4), (20.0, 20.3), (20.3, 20.9))
    ops = [op("same", "E0003", 0, 1), op("absent", "E0004"), op("changed", "E0006", 3, 4), op("absent", "E0009"),
           op("added", None, 2, 2)]
    items = timeline.build_timeline(SCRIPT, ops, asr, 100.0)
    assert [it["id"] for it in items] == [f"I{n:04d}" for n in range(1, 12)]
    assert [it["src_el"] for it in items] == ["E0001", "E0002", "E0003", "E0004", "E0005", None, "E0006", "E0007",
                                              "E0008", "E0009", "E0010"]
    by = {it["src_el"]: it for it in items}
    d = by["E0003"]
    assert (d["type"], d["status"], d["timing"], d["start"], d["end"], d["heard"], d["op"]) == \
        ("dialogue", "same", "aligned", 10.0, 11.0, "w0 w1", 0)
    assert d["words"] == [{"w": "w0", "s": 10.0, "e": 10.5, "p": 0.9}, {"w": "w1", "s": 10.5, "e": 11.0, "p": 0.9}]
    a = by["E0004"]
    assert (a["status"], a["timing"], a["start"], a["end"], a["heard"], a["words"]) == ("absent", "none", None, None, None, [])
    x = by[None]
    assert (x["status"], x["start"], x["end"], x["scene"], x["op"]) == ("added", 12.0, 12.4, "SC001", 4)
    for el in ("E0001", "E0002", "E0005", "E0007", "E0008", "E0010"):
        assert by[el]["timing"] == "estimated" and by[el]["end"] is None and by[el]["op"] is None
        assert "status" not in by[el] and "_weight" not in by[el]
    starts = [it["start"] for it in items if it["start"] is not None]
    assert starts == sorted(starts)


def test_estimated_times_split_by_text_length():
    """대사 아닌 요소는 앞뒤 대사 사이를 글 길이에 비례해 나눈다. 맨 앞은 0, 맨 뒤는 영상 길이가 끝점."""
    sc = S.tiny_script([("1", "scene", "가" * 10), ("1", "action", "나" * 30), ("1", "dialogue", "하나."),
                        ("1", "action", "다" * 10), ("1", "action", "라" * 30), ("1", "dialogue", "둘."),
                        ("1", "action", "마" * 5), ("1", "transition", "바" * 5)])
    asr = words((40.0, 41.0), (61.0, 62.0))
    items = timeline.build_timeline(sc, [op("same", "E0003", 0, 0), op("same", "E0006", 1, 1)], asr, 100.0)
    t = {it["src_el"]: it["start"] for it in items}
    assert (t["E0001"], t["E0002"]) == (0.0, 10.0)                  # 0 ~ 40 을 10:30 으로
    assert (t["E0004"], t["E0005"]) == (41.0, 46.0)                 # 41 ~ 61 을 10:30 으로
    assert (t["E0007"], t["E0008"]) == (62.0, 81.0)                 # 62 ~ 100 을 5:5 로


def test_moved_chunk_boundary_attaches_to_its_scene():
    """앞뒤 대사의 시각이 뒤집힌 곳(옮겨진 덩어리의 경계)에서는 나누지 않고 같은 씬 쪽에 붙인다."""
    sc = S.tiny_script([("1", "scene", "씬1"), ("1", "dialogue", "하나."), ("1", "dialogue", "둘."), ("1", "action", "지문1"),
                        ("2", "scene", "씬2"), ("2", "dialogue", "셋."), ("2", "action", "지문2"),
                        ("3", "scene", "씬3"), ("3", "dialogue", "넷."), ("3", "dialogue", "다섯.")])
    asr = words((100.0, 101.0), (102.0, 103.0), (20.0, 21.0), (110.0, 111.0), (112.0, 113.0))
    ops = [op("same", "E0002", 0, 0), op("same", "E0003", 1, 1), op("same", "E0006", 2, 2), op("same", "E0009", 3, 3),
           op("same", "E0010", 4, 4)]
    items = timeline.build_timeline(sc, ops, asr, 200.0)
    by = {it["src_el"]: it for it in items}
    assert by["E0006"]["moved"] and not by["E0002"]["moved"] and not by["E0009"]["moved"]
    assert by["E0004"]["start"] == 103.0                            # 씬1 지문: 씬1 마지막 대사 끝
    assert by["E0005"]["start"] == 20.0                             # 씬2 머리: 옮겨진 씬2 대사 시작
    assert by["E0007"]["start"] == 21.0                             # 씬2 지문: 씬2 대사 끝
    assert by["E0008"]["start"] == 110.0                            # 씬3 머리: 씬3 첫 대사 시작
    # 씬1 머리는 씬1 의 대사 덩어리 쪽으로 간다: 맨 앞에 끼어든 씬2 덩어리(머리·대사·지문)를 가로지르지 않는다
    assert by["E0001"]["start"] == 21.0
    assert [it["src_el"] for it in items] == ["E0005", "E0006", "E0007", "E0001", "E0002", "E0003", "E0004", "E0008",
                                              "E0009", "E0010"]
    assert [by[e]["near"] for e in ("E0001", "E0004", "E0005", "E0007", "E0008")] == \
        ["E0002", "E0003", "E0006", "E0006", "E0009"]
    spans = {s["id"]: s for s in timeline.scene_spans(sc, items, 200.0)}
    assert (spans["SC002"]["start"], spans["SC002"]["end"]) == (20.0, 21.0)
    assert (spans["SC001"]["start"], spans["SC001"]["end"]) == (100.0, 103.0)


def _film(order, gap=20.0):
    """씬 넷(머리·지문·대사 둘씩)과 대사 없는 씬 5 를 order 차례로 말한 영화. (script, items)."""
    turns = []
    for n in (1, 2, 3, 4):
        turns += [(str(n), "scene", f"S#{n}"), (str(n), "action", f"지문{n}"), (str(n), "dialogue", f"대사{n}가."),
                  (str(n), "dialogue", f"대사{n}나.")]
    turns += [("5", "scene", "S#5"), ("5", "action", "아무도 없다.")]
    sc = S.tiny_script(turns)
    said, ops, t = [], [], 0.0
    for n in order:
        t += gap
        for k in (2, 3):
            el = f"E{(n - 1) * 4 + k + 1:04d}"
            ops.append(op("same", el, len(said), len(said)))
            said.append((t, t + 2.0))
            t += 3.0
    return sc, timeline.build_timeline(sc, ops, words(*said), t + 40.0)


def _blocks(items):
    """시각 있는 항목의 씬을 차례대로, 잇달아 같은 것은 하나로."""
    out = []
    for it in items:
        if it["start"] is not None and (not out or out[-1] != it["scene"]):
            out.append(it["scene"])
    return out


@pytest.mark.parametrize("order", [(1, 2, 3, 4), (1, 3, 4, 2), (1, 3, 2, 4), (2, 1, 3, 4), (3, 1, 2, 4), (1, 4, 2, 3)])
def test_scene_heading_and_action_stay_with_their_dialogue_when_a_scene_is_moved(order):
    """씬이 자리를 옮겨도 한 씬의 씬 머리·지문·대사는 차례에서 한 덩어리다(끼어든 씬의 양쪽으로 갈리지 않는다)."""
    sc, items = _film(order)
    blocks = _blocks(items)
    spoken = [b for b in blocks if b != "SC005"]
    assert spoken == [f"SC{n:03d}" for n in order]                  # 씬마다 한 덩어리, 말한 차례대로
    assert blocks.count("SC005") == 1                               # 대사 없는 씬도 갈리지 않는다
    for n in order:                                                 # 덩어리 안 차례: 씬 머리, 지문, 대사 둘
        own = [it["type"] for it in items if it["scene"] == f"SC{n:03d}"]
        assert own == ["scene", "action", "dialogue", "dialogue"]
    starts = [it["start"] for it in items]
    assert starts == sorted(starts)


def test_estimated_items_do_not_cross_a_moved_chunk_that_abuts_the_previous_line():
    """앞 대사가 끝난 바로 그 시각에 다른 씬의 옮겨진 대사가 시작해도(받아쓰기 낱말은 틈 없이 붙는 일이 많다) 앞 씬의
    지문은 앞 대사 뒤, 옮겨진 씬의 머리는 그 대사 앞이다. 요소 번호 순으로 풀면 옮겨진 씬 머리가 먼저 온다."""
    sc = S.tiny_script([("1", "scene", "S#1"), ("1", "dialogue", "하나."), ("1", "action", "지문1"),
                        ("2", "scene", "S#2"), ("2", "dialogue", "둘."), ("2", "action", "지문2"),
                        ("3", "scene", "S#3"), ("3", "dialogue", "셋."), ("3", "action", "지문3"),
                        ("4", "scene", "S#4"), ("4", "dialogue", "넷.")])
    # 영화 차례: 씬2, 씬3, 씬1(옮겨짐), 씬4. 씬3 대사 끝 = 씬1 대사 시작 = 30.0, 씬1 대사 끝 = 씬4 대사 시작 = 32.0
    asr = words((30.0, 32.0), (10.0, 12.0), (20.0, 30.0), (32.0, 34.0))
    ops = [op("same", "E0002", 0, 0), op("same", "E0005", 1, 1), op("same", "E0008", 2, 2), op("same", "E0011", 3, 3)]
    items = timeline.build_timeline(sc, ops, asr, 100.0)
    assert {it["src_el"] for it in items if it.get("moved")} == {"E0002"}
    order = [it["src_el"] for it in items]
    assert order == ["E0004", "E0005", "E0006", "E0007", "E0008", "E0009", "E0001", "E0002", "E0003", "E0010", "E0011"]
    assert _blocks(items) == ["SC002", "SC003", "SC001", "SC004"]


def test_added_line_takes_the_scene_of_the_nearer_script_line():
    """added 의 씬은 시각으로 앞뒤 대본 대사의 씬 가운데 가까운 쪽이다. 씬 머리·지문의 추정 시각에 따라 갈리지 않는다."""
    sc = S.tiny_script([("1", "scene", "S#1"), ("1", "dialogue", "하나."),
                        ("2", "scene", "S#2"), ("2", "action", "지문"), ("2", "dialogue", "둘."), ("2", "dialogue", "셋.")])

    def scenes_of(*added, dur=300.0):
        said = [(100.0, 101.0), (200.0, 201.0), (203.0, 204.0)] + list(added)
        ops = [op("same", "E0002", 0, 0), op("same", "E0005", 1, 1), op("same", "E0006", 2, 2)]
        ops += [op("added", None, k, k) for k in range(3, len(said))]
        items = timeline.build_timeline(sc, ops, words(*said), dur)
        return [it["scene"] for it in sorted((x for x in items if x["src_el"] is None), key=lambda x: x["start"])], items

    # 앞 대사 끝에서 0.7초 뒤: 앞 씬. 다음 씬 머리의 추정 시각은 앞 대사의 끝(101.0)이 아니라 그 말이 끝난 뒤로 밀린다
    got, items = scenes_of((101.7, 102.3))
    assert got == ["SC001"]
    ids = [it["src_el"] for it in items]
    assert ids.index("E0002") < ids.index(None) < ids.index("E0003")
    assert next(it for it in items if it["src_el"] == "E0003")["start"] == 102.3
    # 뒤 대사 시작 바로 앞: 뒤 씬, 그 씬 머리 뒤에 놓인다
    got, items = scenes_of((198.0, 199.5))
    assert got == ["SC002"]
    ids = [it["src_el"] for it in items]
    assert ids.index("E0003") < ids.index(None) < ids.index("E0005")
    assert next(it for it in items if it["src_el"] == "E0003")["start"] == 101.0
    # 같은 씬의 두 대사 사이: 그 씬
    assert scenes_of((201.5, 202.5))[0] == ["SC002"]
    # 한가운데보다 앞이면 앞 씬, 뒤면 뒤 씬(쉼의 길이로 잰다)
    assert scenes_of((140.0, 141.0))[0] == ["SC001"] and scenes_of((165.0, 166.0))[0] == ["SC002"]
    # 앞 대사에 잇달아 붙어 나온 말들은 길게 이어져도 앞 씬에 남는다(말한 시간은 거리로 세지 않는다)
    chain = [(101.2, 130.0), (130.2, 160.0), (160.3, 185.0)]
    assert scenes_of(*chain)[0] == ["SC001"] * 3
    assert scenes_of(*chain, (199.0, 199.8))[0] == ["SC001"] * 3 + ["SC002"]
    # 맨 앞(앞 대사 없음)은 뒤 대사의 씬, 맨 뒤(뒤 대사 없음)는 앞 대사의 씬
    assert scenes_of((5.0, 6.0), (250.0, 251.0))[0] == ["SC001", "SC002"]
    # 시각 있는 대본 대사가 하나도 없으면 씬을 모른다
    items = timeline.build_timeline(sc, [op("absent", "E0002"), op("absent", "E0005"), op("absent", "E0006"),
                                         op("added", None, 0, 0)], words((5.0, 6.0)), 100.0)
    assert [it["scene"] for it in items if it["src_el"] is None] == [None]


def test_scene_heading_comes_before_the_added_lines_of_its_scene():
    """앞 씬의 긴 지문 때문에 글 길이로 나눈 씬 머리 시각이 늦어져도, 그 씬으로 분류한 added 줄보다 뒤에 놓이지 않는다."""
    sc = S.tiny_script([("1", "scene", "S#1"), ("1", "dialogue", "하나."), ("1", "action", "긴 지문 " * 20),
                        ("2", "scene", "S#2"), ("2", "action", "지문"), ("2", "dialogue", "둘.")])
    ops = [op("same", "E0002", 0, 0), op("same", "E0006", 1, 1), op("added", None, 2, 2), op("added", None, 3, 3)]
    asr = words((100.0, 101.0), (200.0, 201.0), (102.0, 103.0), (160.0, 161.0))
    items = timeline.build_timeline(sc, ops, asr, 300.0)
    by = {it["src_el"]: it for it in items if it["src_el"]}
    first, second = [it for it in items if it["src_el"] is None]
    assert (first["scene"], second["scene"]) == ("SC001", "SC002")
    # 글 길이로만 나누면 씬2 머리는 197초쯤(지문 99자 뒤)이다. 씬2 쪽 말이 시작하는 160초로 당긴다
    assert by["E0004"]["start"] == 160.0 and by["E0003"]["start"] == 101.0
    assert [it["src_el"] for it in items] == ["E0001", "E0002", "E0003", None, "E0004", None, "E0005", "E0006"]
    assert by["E0005"]["start"] > 161.0                              # 씬2 의 지문은 그 뒤 어디든(같은 씬 안)
    assert [(s["id"], s["start"], s["end"]) for s in timeline.scene_spans(sc, items, 300.0)] == \
        [("SC001", 100.0, 103.0), ("SC002", 160.0, 201.0)]


def test_heading_of_a_moved_scene_comes_before_the_added_lines_of_that_scene():
    """옮겨진 씬의 머리는 그 씬의 대사에 붙이되, 그 대사 바로 앞에 붙어 나온 added(같은 씬으로 분류)보다 먼저 놓는다."""
    sc = S.tiny_script([("1", "scene", "S#1"), ("1", "dialogue", "하나."), ("2", "scene", "S#2"), ("2", "dialogue", "둘."),
                        ("3", "scene", "S#3"), ("3", "dialogue", "셋."), ("3", "dialogue", "넷.")])
    #        하나          둘(씬2: 맨 뒤로 옮겨짐)  셋           넷            added: 둘 바로 앞, 넷 바로 뒤
    said = [(10.0, 11.0), (60.0, 61.0), (30.0, 31.0), (33.0, 34.0), (59.0, 59.8), (34.2, 35.0)]
    ops = [op("same", "E0002", 0, 0), op("same", "E0004", 1, 1), op("same", "E0006", 2, 2), op("same", "E0007", 3, 3),
           op("added", None, 4, 4), op("added", None, 5, 5)]
    items = timeline.build_timeline(sc, ops, words(*said), 100.0)
    by = {it["src_el"]: it for it in items if it["src_el"]}
    assert by["E0004"]["moved"] and by["E0003"]["start"] == 59.0 and by["E0005"]["start"] == 30.0
    assert [(it["src_el"], it["scene"]) for it in items] == [
        ("E0001", "SC001"), ("E0002", "SC001"), ("E0005", "SC003"), ("E0006", "SC003"), ("E0007", "SC003"), (None, "SC003"),
        ("E0003", "SC002"), (None, "SC002"), ("E0004", "SC002")]


def test_added_lines_follow_a_moved_line_into_its_scene_and_stay_out_of_the_span():
    """옮겨진 대사 곁의 added 는 그 대사의 씬이지만, 씬 구간은 흐름 안 대사와 거기 딸린 added 로만 잡는다."""
    sc = S.tiny_script([("1", "scene", "S#1"), ("1", "dialogue", "하나."), ("1", "dialogue", "둘."),
                        ("2", "scene", "S#2"), ("2", "dialogue", "셋."), ("2", "dialogue", "넷.")])
    #        하나          둘(옮겨짐: 씬2 뒤)   셋            넷           added 넷: 하나 뒤, 넷 뒤, 둘 앞, 둘 뒤
    said = [(10.0, 11.0), (60.0, 61.0), (30.0, 31.0), (33.0, 34.0), (11.3, 12.0), (34.2, 35.0), (59.0, 59.8), (61.2, 62.0)]
    ops = [op("same", "E0002", 0, 0), op("same", "E0003", 1, 1), op("same", "E0005", 2, 2), op("same", "E0006", 3, 3)]
    ops += [op("added", None, k, k) for k in range(4, 8)]
    items = timeline.build_timeline(sc, ops, words(*said), 100.0)
    extra = sorted((it for it in items if it["src_el"] is None), key=lambda it: it["start"])
    assert [(it["scene"], it["near"]) for it in extra] == [("SC001", "E0002"), ("SC002", "E0006"), ("SC001", "E0003"),
                                                           ("SC001", "E0003")]
    assert not any(it["moved"] for it in extra)                     # moved 는 대본 대사에만 붙는 표시
    spans = {s["id"]: (s["start"], s["end"]) for s in timeline.scene_spans(sc, items, 100.0)}
    assert spans == {"SC001": (10.0, 12.0), "SC002": (30.0, 35.0)}


def test_start_is_strictly_before_end():
    """낱말 하나짜리 줄에서 시작과 끝이 같으면 끝을 0.02초 늘리고 낱말의 e 도 맞춘다(D2)."""
    sc = S.tiny_script([("1", "dialogue", "예.")])
    items = timeline.build_timeline(sc, [op("same", "E0001", 0, 0)], words((5.0, 5.0)), 10.0)
    it = items[0]
    assert it["start"] == 5.0 and it["end"] == 5.02 and it["words"][0]["e"] == 5.02
    asr = S.asr_of(S.say([("예", 5.0, 5.0), ("", 5.0, 5.0), ("네.", 5.0, 5.0)]))
    it = timeline.build_timeline(sc, [op("changed", "E0001", 0, 2)], asr, 10.0)[0]
    assert it["heard"] == "예 네." and len(it["words"]) == 2      # 글자 없는 낱말은 뺀다
    assert it["start"] < it["end"] and all(it["start"] <= w["s"] <= w["e"] <= it["end"] for w in it["words"])
    assert it["time_fix"] == {"fixed": 1, "back": 0.0, "late": 0}


def _fix(*spans, duration=None):
    ws, info = timeline.fix_words([{"i": i, "w": f"w{i}", "s": s, "e": e, "p": 0.9} for i, (s, e) in enumerate(spans)],
                                  duration)
    return [(w["s"], w["e"]) for w in ws], info


def test_word_times_are_repaired_without_touching_text_or_order():
    """낱말 시각 바로잡기: 시작이 앞 낱말 시작보다 이르면 앞 낱말 시작으로, 끝은 시작보다 이르지 않게. 글자·차례는 그대로."""
    # 멀쩡한 낱말은 그대로(겹쳐도, 틈이 있어도, 시작이 같아도)
    clean = [(10.0, 10.5), (10.5, 11.0), (10.9, 11.4), (13.0, 13.2), (13.0, 13.6)]
    assert _fix(*clean) == (clean, {"fixed": 0, "back": 0.0, "late": 0})
    # 낱말 하나가 앞 낱말보다 0.05초 먼저 시작
    assert _fix((10.0, 10.5), (9.95, 10.2), (10.6, 10.9)) == \
        ([(10.0, 10.5), (10.0, 10.2), (10.6, 10.9)], {"fixed": 1, "back": 0.05, "late": 0})
    # 받아쓰기 묶음이 겹쳐 같은 말이 0.6초 앞 시각으로 한 번 더 적힘: 끝이 시작보다 이르면 시작에 맞춘다
    assert _fix((10.0, 10.5), (9.4, 9.9), (10.5, 11.0)) == \
        ([(10.0, 10.5), (10.0, 10.0), (10.5, 11.0)], {"fixed": 1, "back": 0.6, "late": 0})
    # 올린 시작이 다음 낱말의 바닥이 된다(잇달아 뒤집힌 낱말)
    got, info = _fix((10.0, 10.5), (9.0, 9.2), (9.5, 9.7), (10.2, 10.4))
    assert got == [(10.0, 10.5), (10.0, 10.0), (10.0, 10.0), (10.2, 10.4)] and info == {"fixed": 2, "back": 1.0, "late": 0}
    assert not timeline.suspect(info) and timeline.suspect({"fixed": 1, "back": 1.2, "late": 0})
    # 글자와 차례, 확률은 건드리지 않는다
    ws, _ = timeline.fix_words([{"i": 7, "w": "가", "s": 5.0, "e": 5.4, "p": 0.3}, {"i": 8, "w": "나", "s": 4.0, "e": 4.2, "p": None}])
    assert ws == [{"w": "가", "s": 5.0, "e": 5.4, "p": 0.3}, {"w": "나", "s": 5.0, "e": 5.0, "p": None}]


def test_word_times_past_the_video_are_pulled_to_its_end():
    """영상 길이를 0.5초 넘게 벗어난 시각만 영상 끝으로 당긴다(영상 길이를 주었을 때). 그런 줄은 믿기 어려운 줄로 표시."""
    assert _fix((599.0, 599.8), (599.8, 600.4), duration=600.0) == \
        ([(599.0, 599.8), (599.8, 600.4)], {"fixed": 0, "back": 0.0, "late": 0})          # 허용 오차 안
    got, info = _fix((599.0, 599.8), (599.9, 601.0), (602.0, 603.0), duration=600.0)
    assert got == [(599.0, 599.8), (599.9, 600.0), (600.0, 600.0)] and info == {"fixed": 2, "back": 0.0, "late": 2}
    assert timeline.suspect(info)
    got, info = _fix((700.0, 701.0), duration=600.0)                                       # 줄 전체가 영상 밖
    assert got == [(600.0, 600.02)] and info["late"] == 1 and info["fixed"] == 1
    # 허용 오차 끝에 걸린 한 순간짜리 줄: 0.02초 늘려도 영상 길이 + 0.5초를 넘지 않는다
    for t in (600.44, 600.46, 600.47, 600.5, 600.51):
        (s, e), = _fix((t, t), duration=600.0)[0]
        assert s < e <= 600.5 and (s, e) == ((t, round(t + 0.02, 3)) if t <= 600.46 else (600.0, 600.02))
    assert _fix((700.0, 701.0))[0] == [(700.0, 701.0)]                                     # 영상 길이를 모르면 그대로
    # build_timeline 은 영상 길이를 준다: 항목 시각이 영상 길이 + 허용 오차를 넘지 않는다
    sc = S.tiny_script([("1", "dialogue", "하나."), ("1", "dialogue", "둘.")])
    asr = words((598.0, 599.0), (600.2, 601.5), (601.5, 602.0))
    items = timeline.build_timeline(sc, [op("same", "E0001", 0, 0), op("same", "E0002", 1, 2)], asr, 600.0)
    last = items[1]
    assert (last["start"], last["end"]) == (600.0, 600.02) and last["time_fix"] == {"fixed": 2, "back": 0.0, "late": 2}
    assert items[0]["time_fix"] == {"fixed": 0, "back": 0.0, "late": 0}
    index = timeline.asr_index(asr)
    assert timeline.op_words(index, op("same", "E0002", 1, 2), 600.0) == last["words"]
    assert [w["s"] for w in timeline.op_words(index, op("same", "E0002", 1, 2))] == [600.2, 601.5]
    # 허용 오차 안에서 영상 길이를 넘은 줄: 낱말도 씬 구간도 그 시각 그대로(구간을 영상 길이로 잘라 항목과 어긋나게 하지 않는다)
    sc2 = S.tiny_script([("1", "dialogue", "하나."), ("2", "scene", "S#2"), ("2", "dialogue", "둘.")])
    items = timeline.build_timeline(sc2, [op("same", "E0001", 0, 0), op("same", "E0003", 1, 1)],
                                    words((598.0, 599.0), (600.2, 600.4)), 600.0)
    assert (items[-1]["start"], items[-1]["end"], items[-1]["time_fix"]["fixed"]) == (600.2, 600.4, 0)
    assert timeline.scene_spans(sc2, items, 600.0)[1] == {"id": "SC002", "start": 600.2, "end": 600.4, "timing": "estimated"}


def test_reversed_word_times_inside_a_line_do_not_break_the_timeline():
    """한 줄 안에서 낱말 시작 시각이 거꾸로 가도 항목은 start < end, 낱말은 시각 차례·줄 범위 안이다(final 의 D2)."""
    sc = S.tiny_script([("1", "dialogue", "하나 둘 셋."), ("1", "dialogue", "넷.")])
    asr = words((10.0, 10.4), (9.7, 10.1), (10.4, 10.9), (12.0, 12.5))
    items = timeline.build_timeline(sc, [op("changed", "E0001", 0, 2), op("same", "E0002", 3, 3)], asr, 60.0)
    it = items[0]
    assert [(w["w"], w["s"], w["e"]) for w in it["words"]] == [("w0", 10.0, 10.4), ("w1", 10.0, 10.1), ("w2", 10.4, 10.9)]
    assert (it["start"], it["end"], it["heard"]) == (10.0, 10.9, "w0 w1 w2")
    assert it["time_fix"] == {"fixed": 1, "back": 0.3, "late": 0} and items[1]["time_fix"]["fixed"] == 0


def _ops(*ranges):
    return [{"item": f"I{k:04d}", "asr": [f"{p}:{a}", f"{p}:{b}"]} for k, (p, a, b) in enumerate(ranges, 1)]


def test_shared_words_reports_a_word_held_by_two_ops():
    """같은 pass 의 낱말 범위가 겹치면(같은 낱말이 두 op 에) 알린다. 글 없이 항목 id 와 낱말 번호만."""
    assert timeline.shared_words(_ops(("plain", 0, 1), ("plain", 2, 3), ("lv3", 0, 3), ("rescue", 2, 2))) == []
    got = timeline.shared_words(_ops(("plain", 0, 2), ("plain", 2, 3)))            # 낱말 plain:2 가 두 줄에
    assert got == ["I0001 와 I0002 가 같은 낱말을 가짐(plain:2~2)"]
    got = timeline.shared_words(_ops(("rescue", 1, 2), ("plain", 5, 9), ("rescue", 0, 4)))   # 한 범위가 다른 범위 안에
    assert got == ["I0003 와 I0001 가 같은 낱말을 가짐(rescue:1~2)"]
    assert len(timeline.shared_words(_ops(("plain", 0, 5), ("plain", 1, 2), ("plain", 4, 6)))) == 2
    # 범위가 없거나 깨진 op 은 건너뛴다(다른 검사가 알린다)
    assert timeline.shared_words([{"item": "I0001", "asr": None}, {"item": "I0002", "asr": ["plain:3", "plain:1"]},
                                  {"item": "I0003", "asr": ["plain:1", "ctx:2"]}, {"asr": ["plain:0", "plain:0"]}]) == []


def test_cross_pass_overlaps_reports_lines_from_two_passes_over_the_same_time():
    """서로 다른 pass 에서 온 두 줄의 시간 구간이 0.6초 넘게 겹치면 알린다. pass 사이 낱말 경계의 흔들림은 걸리지 않는다."""
    plain = S.say([("가", 10.0, 10.4), ("나", 10.4, 10.8), ("다", 10.8, 11.2), ("라", 12.0, 12.4)])
    lv3 = S.say([("가", 9.9, 10.3), ("나", 10.3, 10.7), ("다아", 10.7, 11.25), ("라", 11.9, 12.4), ("마", 12.5, 12.9)], "lv3")
    index = timeline.asr_index(S.asr_of(plain, lv3))

    def over(*ranges):
        return timeline.cross_pass_overlaps(_ops(*ranges), index)

    assert over(("plain", 0, 1), ("plain", 2, 3)) == []                              # 한 pass 뿐
    assert over(("plain", 0, 1), ("lv3", 2, 4)) == []                                # 경계가 0.1초 겹침
    assert over(("lv3", 0, 2), ("plain", 3, 3)) == [] and over(("lv3", 0, 1), ("plain", 2, 3)) == []
    assert over(("plain", 0, 2), ("lv3", 2, 3)) == []                                # 낱말 하나(0.5초)가 두 줄에: 시각만으로는 모른다
    assert over(("plain", 0, 2), ("lv3", 1, 4)) == [("I0001", "I0002", 0.9)]         # 두 낱말(0.9초)이 두 줄에
    assert over(("plain", 0, 3), ("lv3", 0, 4)) == [("I0002", "I0001", 2.4)]
    assert over(("plain", 0, 3), ("lv3", 0, 4), ("plain", 9, 9)) == [("I0002", "I0001", 2.4)]   # 없는 범위는 건너뛴다
    assert timeline.cross_pass_overlaps(_ops(("plain", 0, 2), ("lv3", 1, 4)), index, 1.0) == []
    both = timeline.double_claims(_ops(("plain", 0, 2), ("plain", 2, 3), ("lv3", 0, 4)), index)
    assert len(both) == 3 and "같은 낱말" in both[0] and "서로 다른 pass" in both[1] and "1.200초" in both[1]


def test_words_keep_final_shape():
    """낱말은 w, s, e, p 만(번호 i 는 뗀다). 시각은 소수 셋째 자리."""
    sc = S.tiny_script([("1", "dialogue", "예.")])
    asr = S.asr_of({"id": "plain", "model": "m", "context": None,
                    "words": [{"i": 0, "w": " 예. ", "s": 1.00049, "e": 1.5, "p": None, "lang": "ko"}]})
    it = timeline.build_timeline(sc, [op("same", "E0001", 0, 0)], asr, 10.0)[0]
    assert it["words"] == [{"w": "예.", "s": 1.0, "e": 1.5, "p": None}]


def test_ops_must_cover_every_dialogue_element_once():
    """대본 대사 요소마다 op 이 정확히 하나여야 한다."""
    sc = S.tiny_script([("1", "dialogue", "하나."), ("1", "dialogue", "둘.")])
    asr = words((1.0, 2.0))
    with pytest.raises(ValueError):
        timeline.build_timeline(sc, [op("same", "E0001", 0, 0)], asr, 10.0)
    with pytest.raises(ValueError):
        timeline.build_timeline(sc, [op("same", "E0001", 0, 0), op("absent", "E0001"), op("absent", "E0002")], asr, 10.0)
    with pytest.raises(ValueError):
        timeline.build_timeline(sc, [op("same", "E0001", 0, 0), op("absent", "E0002"), op("absent", "E0009")], asr, 10.0)
    with pytest.raises(ValueError):
        timeline.build_timeline(sc, [op("absent", "E0001"), op("absent", "E0002"), op("same", None, 0, 0)], asr, 10.0)


def test_front_and_unknown_elements_are_left_out():
    """front·unknown 요소는 항목에 넣지 않는다. 나머지 요소는 정확히 한 번씩 나온다."""
    sc = S.tiny_script([("1", "scene", "씬"), ("1", "dialogue", "하나.")])
    sc["elements"].insert(0, {"id": "E0000", "type": "front", "scene": None, "text": "표지", "lines": [0]})
    sc["elements"].append({"id": "E0003", "type": "unknown", "scene": "SC001", "text": "?", "lines": [3]})
    items = timeline.build_timeline(sc, [op("absent", "E0002")], words(), 10.0)
    assert [it["src_el"] for it in items] == ["E0001", "E0002"]


def test_attach_items_writes_ids_into_ops():
    asr = words((10.0, 10.5), (10.5, 11.0), (12.0, 12.4), (20.0, 20.3), (20.3, 20.9))
    ops = [op("same", "E0003", 0, 1), op("absent", "E0004"), op("changed", "E0006", 3, 4), op("absent", "E0009"),
           op("added", None, 2, 2)]
    items = timeline.build_timeline(SCRIPT, ops, asr, 100.0)
    timeline.attach_items(ops, items)
    assert [o["item"] for o in ops] == ["I0003", "I0004", "I0007", "I0010", "I0006"]
    with pytest.raises(ValueError):
        timeline.attach_items(ops + [op("added", None, 2, 2)], items)


def test_scene_spans():
    """씬 구간: 대사 시각이 있으면 그 범위, 대사가 없거나 다 absent 면 추정 시각 범위, 요소가 없으면 none."""
    asr = words((10.0, 10.5), (10.5, 11.0), (12.0, 12.4), (20.0, 20.3), (20.3, 20.9))
    ops = [op("same", "E0003", 0, 1), op("absent", "E0004"), op("changed", "E0006", 3, 4), op("absent", "E0009"),
           op("added", None, 2, 2)]
    sc = dict(SCRIPT, scenes=SCRIPT["scenes"] + [{"id": "SC003", "number": "3", "heading": "빈 씬", "place": None,
                                                   "time": None, "flags": []}])
    items = timeline.build_timeline(sc, ops, asr, 100.0)
    spans = timeline.scene_spans(sc, items, 100.0)
    assert spans[0] == {"id": "SC001", "start": 10.0, "end": 20.9, "timing": "estimated"}
    assert spans[1]["timing"] == "estimated" and 20.9 <= spans[1]["start"] <= spans[1]["end"] <= 100.0
    assert spans[2] == {"id": "SC003", "start": None, "end": None, "timing": "none"}


def _random_case(rng: random.Random, durs=(0.0, 0.3, 1.2)):
    """무작위 영화: 줄 일부는 absent, 덩어리 몇 개는 자리를 옮기고, added 를 섞는다. durs: 줄 길이(초) 후보."""
    n = rng.randint(5, 40)
    turns = []
    for k in range(n):
        if rng.random() < 0.25:
            turns.append((str(1 + k // 6), rng.choice(["scene", "action", "onscreen", "transition"]), f"요소{k}"))
        turns.append((str(1 + k // 6), "dialogue", f"대사{k}."))
    sc = S.tiny_script(turns)
    dl = [e["id"] for e in sc["elements"] if e["type"] == "dialogue"]
    order = list(range(len(dl)))
    for _ in range(rng.randint(0, 3)):                             # 덩어리 옮기기
        a = rng.randrange(len(order))
        b = min(len(order), a + rng.randint(1, 4))
        chunk = order[a:b]
        del order[a:b]
        at = rng.randint(0, len(order))
        order[at:at] = chunk
    spans, ops, t = [], [], 1.0
    slot = {}
    for k in order:
        if rng.random() < 0.3:
            continue
        t += rng.choice([0.0, 0.5, 3.0])
        dur = rng.choice(durs)
        slot[k] = len(spans)
        spans.append((round(t, 3), round(t + dur, 3)))
        t += dur
        if rng.random() < 0.2:
            t += 0.4
            spans.append((round(t, 3), round(t + 0.5, 3)))
            ops.append(op("added", None, len(spans) - 1, len(spans) - 1))
            t += 0.5
    for k, el in enumerate(dl):
        ops.append(op("same", el, slot[k], slot[k]) if k in slot else op("absent", el))
    rng.shuffle(ops)
    return sc, ops, words(*spans), t + 5.0


def test_untimed_items_pass_validate_d7_on_random_films():
    """덩어리를 무작위로 옮기고 added·absent 를 섞어도 차례가 validate 의 D7 검사를 통과한다."""
    rng = random.Random(7)
    for _ in range(300):
        sc, ops, asr, dur = _random_case(rng)
        items = timeline.build_timeline(sc, ops, asr, dur)
        errs = validate._Errs()
        validate._check_untimed_order(items, errs)
        assert errs.done() == []
        starts = [it["start"] for it in items if it["start"] is not None]
        assert starts == sorted(starts)                             # 시각 있는 항목은 start 가 줄지 않는다
        assert [it["id"] for it in items] == [f"I{k:04d}" for k in range(1, len(items) + 1)]
        want = sorted(e["id"] for e in sc["elements"])
        assert sorted(it["src_el"] for it in items if it["src_el"]) == want     # 요소마다 정확히 한 번
        assert sum(it["src_el"] is None for it in items) == sum(o["op"] == "added" for o in ops)
        for it in items:
            if it["type"] == "dialogue" and it["start"] is not None:
                assert it["start"] < it["end"]
                assert all(it["start"] <= w["s"] <= w["e"] <= it["end"] for w in it["words"])
            if it["start"] is not None:
                assert 0.0 <= it["start"] <= max(dur, it["start"])


def test_items_sit_next_to_the_line_they_belong_to_on_random_films():
    """덩어리를 무작위로 옮기고 added 를 섞어도, 대사에 딸린 항목(지문·씬 머리·added)과 그 대사 사이에는 같은 씬의 항목만
    있다(다른 씬의 대사나 지문이 끼지 않는다). added 의 씬은 시각으로 앞뒤에 있는 대본 대사의 씬 가운데 하나다."""
    rng = random.Random(11)
    for _ in range(400):
        sc, ops, asr, dur = _random_case(rng, durs=(0.3, 1.2))       # 줄끼리 시간이 겹치지 않는 영화
        items = timeline.build_timeline(sc, ops, asr, dur)
        timed = [it for it in items if it["start"] is not None]
        at = {it["src_el"]: k for k, it in enumerate(timed) if it["src_el"]}
        said = [it for it in timed if it["type"] == "dialogue" and it["src_el"]]
        for k, it in enumerate(timed):
            if it["near"] is not None:
                j = at[it["near"]]
                between = timed[min(j, k) + 1:max(j, k)]
                assert all(x["scene"] == it["scene"] for x in between), (it["id"], [x["id"] for x in between])
                assert timed[j]["scene"] == it["scene"] and timed[j]["type"] == "dialogue"
            if it["type"] == "dialogue" and it["src_el"] is None:
                before = [d for d in said if d["start"] <= it["start"]]
                after = [d for d in said if d["start"] > it["start"]]
                around = {d["scene"] for d in before[-1:] + after[:1]}
                assert (it["scene"] in around) if around else it["scene"] is None
        for it in items:
            assert set(it) >= {"id", "type", "src_el", "scene", "start", "end", "timing", "op", "near"}
            assert not any(key.startswith("_") for key in it)


def test_same_input_gives_same_timeline():
    rng = random.Random(3)
    sc, ops, asr, dur = _random_case(rng)
    a = timeline.build_timeline(sc, ops, asr, dur)
    b = timeline.build_timeline(sc, [dict(o) for o in ops], asr, dur)
    assert a == b
