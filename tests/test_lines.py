from sinkprep import lines


def seg(text, x0, y0, x1, y1, **kw):
    return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, **kw}


def test_same_baseline_far_apart_joined_with_tab():
    segs = [seg("형사", 100, 200, 130, 212), seg("너 어제 어디 있었어?", 300, 201, 480, 213)]
    out = lines.merge_segments(segs, 600, 800)
    assert len(out) == 1
    assert out[0]["text"] == "형사\t너 어제 어디 있었어?"
    assert abs(out[0]["x"] - 100 / 600) < 1e-3


def test_close_segments_joined_with_space_and_rows_split_by_y():
    segs = [seg("JOEL", 200, 100, 240, 110), seg("(V.O.)", 245, 100, 290, 110),
            seg("Valentine's Day.", 150, 120, 300, 130)]
    out = lines.merge_segments(segs, 600, 800)
    assert [r["text"] for r in out] == ["JOEL (V.O.)", "Valentine's Day."]
    assert out[0]["y"] < out[1]["y"]


def test_conf_and_bold():
    segs = [seg("S#1. 빵집", 50, 10, 150, 22, conf=0.9, bold=True), seg("낮", 160, 10, 180, 22, conf=0.7, bold=True)]
    out = lines.merge_segments(segs, 600, 800)
    assert out[0]["conf"] == 0.7 and out[0]["bold"] is True


def test_text_lines_indent_and_gaps():
    txt = "S#1. 빵집\n\n\t\t철수\n        안녕?\n"
    out = lines.lines_from_text(txt, "txt")
    assert [r["text"] for r in out] == ["S#1. 빵집", "철수", "안녕?"]
    assert out[0]["x"] == 0 and out[1]["x"] == round(16 / 80, 4) and out[2]["x"] == round(8 / 80, 4)
    assert out[0]["page"] is None
    # 빈 줄은 y 간격으로 남는다
    assert out[1]["y"] - out[0]["y"] > out[2]["y"] - out[1]["y"]


def test_decode_cp949_and_utf8():
    assert lines.decode_text("대본".encode("cp949")) == "대본"
    assert lines.decode_text("대본".encode("utf-8")) == "대본"
    assert lines.decode_text("﻿대본".encode("utf-8")) == "대본"


def test_clean_text_nfc_and_controls():
    import unicodedata
    s = unicodedata.normalize("NFD", "한글") + "\x00\x07 "
    assert lines.clean_text(s) == "한글"


def _ch(c, x, y, w=6.0, h=10.0):
    return {"c": c, "bbox": [x, y, x + w, y + h]}


def test_overprint_bold_kept_once():
    """굵게 보이려고 0.12pt 씩 밀어 여러 번 찍은 '5.' 는 한 번만 남고, 떨어진 같은 글자는 남는다."""
    copies = [{"size": 10.9, "flags": 0, "chars": [_ch("5", 85.0 + 0.12 * i, 98.9 + 0.12 * i),
                                                   _ch(".", 91.3 + 0.12 * i, 98.9 + 0.12 * i)]}
              for i in range(5)]
    other = {"size": 10.9, "flags": 0, "chars": [_ch("5", 300.0, 98.9), _ch(".", 306.0, 98.9)]}
    out = lines._drop_overprint(copies + [other])
    texts = ["".join(c["c"] for c in sp["chars"]) for sp in out]
    assert texts == ["5.", "", "", "", "", "5."]


def test_glyph_map_lookalikes():
    """글자층이 공백을 U+2029, 쌍점을 U+A789, 줄표를 U+2011 로 쓴 경우 보통 글자로."""
    sp = {"size": 10, "flags": 0, "chars": [_ch(c, 10 + 6 * i, 0) for i, c in enumerate("INT.\u2029BAR\ua789\u2011")]}
    assert lines._span_segments(sp)[0]["text"] == "INT. BAR:-"
    assert lines.clean_text("a\u2029b\u00adc") == "a bc"


def _row_span(text, x, y, size=10.0, w=6.0):
    return {"size": size, "flags": 0, "chars": [_ch(c, x + w * i, y) for i, c in enumerate(text)]}


def test_two_up_gutter_split():
    """가로 쪽(842x595)에 대본 두 쪽이 나란히: 가운데 빈 띠에서 나누고, 같은 높이의 왼쪽·오른쪽 글을 잇지 않는다."""
    spans = []
    for k in range(20):
        spans.append(_row_span("왼쪽 쪽의 지문 줄입니다", 60, 60 + 20 * k))
        spans.append(_row_span("오른쪽 쪽의 지문 줄", 480, 60 + 20 * k))
    g = lines._gutter(spans, 842, 595)
    assert g is not None and 60 + 6 * 12 < g < 480
    left, right = lines._halves(spans, g)
    assert all("오른쪽" not in "".join(c["c"] for c in sp["chars"]) for sp in left)
    assert len(left) == 20 and len(right) == 20
    # 세로 쪽이나 한쪽에만 글이 있으면 펼침이 아님
    assert lines._gutter(spans, 595, 842) is None
    assert lines._gutter([s for s in spans if s["chars"][0]["bbox"][0] < 100], 842, 595) is None
