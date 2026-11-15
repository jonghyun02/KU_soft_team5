from pathlib import Path

from sinkprep import lines, util


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


# ---------------------------------------------------------------- 원본 글로 공백·탭 되살리기

def test_respace_restores_spaces_and_tabs():
    """추격자(fm55664) 꼴: 대사 줄의 띄어쓰기가 없고 긴 이름 뒤 탭이 공백이 된 PDF 줄에 원본(HWP 문단)의 공백·탭을 되살린다.
    문단 앞머리 탭(⇥이름⇥대사 의 첫 탭)은 줄 글에 넣지 않는다(들여쓰기는 x 가 나타낸다). 글 맨 앞 문단도 마찬가지다."""
    source = ("\t중호\t씨발년... 넌 잡히면 죽는다.\n"
              "\t모텔남1\t(침대에 누우며) 나이가 좀 많네. 몇 살이야?\n"
              "\t\t예. 지금 보내 드릴께요.\n"
              "\t(영민)\t차를 저렇게 대면 어떡해요?\n"
              "\t모텔남1\t빨리 왔네.\n"
              "\t영민\t.......\n")
    pdf = ["중호\t씨발년...넌잡히면죽는다.", "모텔남1 (침대에누우며)나이가좀많네.몇살이야?", "예.지금보내드릴께요.",
           "(영민) 차를저렇게대면어떡해요?", "모텔남1 빨리왔네.", "영민\t......."]
    out, st = lines.respace(pdf, source)
    assert out == ["중호\t씨발년... 넌 잡히면 죽는다.", "모텔남1\t(침대에 누우며) 나이가 좀 많네. 몇 살이야?",
                   "예. 지금 보내 드릴께요.", "(영민)\t차를 저렇게 대면 어떡해요?", "모텔남1\t빨리 왔네.", "영민\t......."]
    assert st["missing"] == 16 and st["tab_down"] == 3 and st["raised"] == 3 and st["tabs"] == 4
    assert st["loss"] > 0.7 and st["matched"] == 1.0 and st["lines"] == 5
    # 글이 '⇥이름⇥대사' 문단으로 시작해도 첫 문단의 이름 뒤 탭이 되살아난다(앞머리 탭을 이름 뒤 탭으로 세지 않음)
    out, _ = lines.respace(["모텔남1 빨리왔네.", "모텔남1 나이가좀많네."], "\t모텔남1\t빨리 왔네.\n\t모텔남1\t나이가 좀 많네.\n")
    assert out == ["모텔남1\t빨리 왔네.", "모텔남1\t나이가 좀 많네."]


def test_respace_follows_source_spacing():
    """줄 글의 띄어쓰기를 원본에 맞춘다: 빠진 공백은 넣고, 원본에 없는 PDF 공백('암전 되며', '야! ! 우리가')은 지우고, 원본이
    공백 하나인 자리의 탭은 공백으로 내린다. PDF 에만 있는 글자(자동 번호 '6.', 쪽 번호)와 그 옆의 틈은 그대로다."""
    source = ("망원동 골목길 : 오프닝 시퀀스 / 오후-밤\n"
              "화면 암전되며, 타이틀. 추격자.\n"
              "비 내리는 골목길 곳곳의 이미지들이 보여진다. 음침해 보이는 각종 주택들.\n"
              "야!! 우리가 됐어!!\n")
    pdf = ["6.\t망원동 골목길 : 오프닝 시퀀스 / 오후-밤",         # '6.' 은 한글 자동 번호(원본 글에 없음)
           "화면 암전 되며, 타이틀.\t추격자.",                     # 원본에 없는 공백('암전 되며'), 원본은 공백인데 PDF 는 탭
           "- 2 -",                                                # 쪽 번호(원본에 없음)
           "비내리는 골목길곳곳의 이미지들이 보여진다.",           # 한 문단이 두 줄로
           "음침해보이는 각종주택들.",
           "야! ! 우리가 됐어! !"]
    out, st = lines.respace(pdf, source)
    assert out == ["6.\t망원동 골목길 : 오프닝 시퀀스 / 오후-밤",
                   "화면 암전되며, 타이틀. 추격자.",
                   "- 2 -",
                   "비 내리는 골목길 곳곳의 이미지들이 보여진다.",
                   "음침해 보이는 각종 주택들.",
                   "야!! 우리가 됐어!!"]
    assert st["missing"] == 4 and st["spaces"] == 4 and st["removed"] == 3 and st["lowered"] == 1 and st["lines"] == 4
    # 공백을 뺀 글자열은 한 글자도 바뀌지 않고 줄 수도 그대로다
    assert ["".join(t.split()) for t in out] == ["".join(t.split()) for t in pdf]


def test_respace_only_name_head_tab_becomes_tab():
    """문단 머리의 이름 뒤 탭만 탭으로 되살린다. 문단 중간의 탭(줄바꿈 자리를 맞추려고 친 탭, 클래식 fm54441)과 이름 꼴이
    아닌 첫 덩어리 뒤의 탭(씬 번호, 문장)은 넓은 공백으로 본다: PDF 에 틈이 있으면 그대로, 없으면 공백 하나."""
    source = ("아줌마\t...?\t콩나물은 왜 찾아? 여기가 시장이냐,\t슈퍼냐? 그런 걸 가지고\t장난을 치면 되겠어?\n"
              "아줌마\t뭐라구?\n"
              "S# 114\t수하 하숙방 - 저녁\n"
              "전화를 끊고는 수첩을 꺼내드는 봉두.\t학생들 명단 옆에는 전화번호가 있다.\n")
    pdf = ["아줌마 ...? 콩나물은왜찾아?여기가시장이냐,슈퍼냐?그런걸", "가지고 장난을치면되겠어?",       # 한 문단이 두 줄로
           "아줌마 뭐라구?", "S# 114 수하 하숙방 - 저녁", "전화를 끊고는 수첩을 꺼내드는 봉두. 학생들 명단 옆에는 전화번호가 있다."]
    out, st = lines.respace(pdf, source)
    assert out == ["아줌마\t...? 콩나물은 왜 찾아? 여기가 시장이냐, 슈퍼냐? 그런 걸", "가지고 장난을 치면 되겠어?",
                   "아줌마\t뭐라구?", "S# 114 수하 하숙방 - 저녁", "전화를 끊고는 수첩을 꺼내드는 봉두. 학생들 명단 옆에는 전화번호가 있다."]
    assert st["tabs"] == 2 and st["tab_down"] == 2
    for chunk, kind in [("오형사", 1), ("모텔남1", 1), ("(영민)", 1), ("견우(나레이션)", 1), ("제임스 :", 1), ("진 수", 1),
                        ("*김상두 :", 1), ("철용, 미수", 2), ("진태 모", 2), ("S# 114", 0), ("#49.", 0), ("씬 /57", 0),
                        ("12-13.", 0), ("최형사.", 0), ("무슨 일이 일어날 지", 0), ("(T.V의 화면이 이어지며,", 0),
                        ("플래시백, 전시관-", 0)]:
        assert lines._name_head(chunk)[0] == kind, chunk
    assert lines._name_head("감식반2")[1] == lines._name_head("(감식반)")[1] == lines._name_head("감식반 (E)")[1] == "감식반"


def test_respace_name_tab_needs_evidence():
    """이름 뒤 탭은 그 이름이 원본에서 '이름⇥…' 꼴로 두 번 이상 나오거나 줄이 이름 칸에 놓였을 때만 되살린다. 한 번뿐인
    '말들을⇥보며 맥주 한 캔을…'(각설탕 fm55273), '스스로⇥강해질 필요가…'(완득이)는 문장 첫 낱말 뒤에 친 탭이다.
    자동 번호가 앞에 붙은 줄('＃41. 출판사⇥/ 낮')은 줄 머리가 아니고, 낱말 한가운데 친 탭('전 남자분⇥인 줄')도 아니다."""
    paras = []
    for k in range(4):
        paras += [f"\t이선생\t{k}번 문제는 다음 시간에 풉니다.", f"\t지원모\t(웃으며) 그래요, {k}번은 어렵죠."]
    paras += ["\t잠옷아이\t왜요?",                       # 한 번뿐인 단역: 이름 칸에 놓였으면 받는다
              "스스로\t강해질 필요가 있었습니다.",       # 한 번뿐이고 이름 칸이 아님
              "말들을\t보며 맥주 한 캔을 하다보면, 여름밤이",
              "출판사\t/ 낮", "출판사\t/ 밤",            # 자동 번호 씬 머리(원본 글에는 번호가 없다)
              "전 남자분\t인 줄 알았는데.", "전 남자분\t이 오셨어요.",
              "진태 모\t(손잡고) 어..어..", "진태 모\t(울며) 아이고."]
    source = "\n".join(paras) + "\n"
    pdf, xs = [], []
    for k in range(4):
        pdf += [f"이선생 {k}번 문제는 다음 시간에 풉니다.", f"지원모 (웃으며) 그래요, {k}번은 어렵죠."]
        xs += [0.21, 0.21]
    pdf += ["잠옷아이 왜요?", "스스로 강해질 필요가 있었습니다.", "말들을 보며 맥주 한 캔을 하다보면, 여름밤이",
            "＃41. 출판사 / 낮", "＃42. 출판사 / 밤", "전 남자분인 줄 알았는데.", "전 남자분이 오셨어요.",
            "진태 모(손잡고) 어..어..", "진태 모 (울며) 아이고."]
    xs += [0.2105, 0.2774, 0.4118, 0.1429, 0.1429, 0.2774, 0.2774, 0.21, 0.21]
    out, st = lines.respace(pdf, source, xs=xs)
    assert out[:8] == [t.replace(" ", "\t", 1) for t in pdf[:8]]
    # 받지 않은 탭 자리는 넓은 공백으로 본다: PDF 줄에 틈이 있으면 그대로, 없으면 공백 하나('전 남자분 인 줄')
    assert out[8:] == ["잠옷아이\t왜요?", "스스로 강해질 필요가 있었습니다.", "말들을 보며 맥주 한 캔을 하다보면, 여름밤이",
                       "＃41. 출판사 / 낮", "＃42. 출판사 / 밤", "전 남자분 인 줄 알았는데.", "전 남자분 이 오셨어요.",
                       "진태 모\t(손잡고) 어..어..", "진태 모\t(울며) 아이고."]
    assert st["raised"] == 11 and st["tab_missing"] == 1 and st["spaces"] == 2
    # 줄의 x 를 모르면 이름 칸을 볼 수 없어 한 번뿐인 이름은 받지 않는다
    out, _ = lines.respace(pdf, source)
    assert out[8] == "잠옷아이 왜요?" and out[0] == "이선생\t0번 문제는 다음 시간에 풉니다."


def test_respace_does_not_join_across_source_paragraphs():
    """PDF 한 줄이 원본의 두 문단에 걸치면(같은 높이의 두 칸) 그 사이에는 아무것도 넣지 않는다."""
    out, _ = lines.respace(["가나다라마바사아자차카타파하"], "가나다라 마바사\n아자차카 타파하\n")
    assert out == ["가나다라 마바사아자차카 타파하"]


def test_respace_ignores_chance_matches_between_unrelated_text(monkeypatch):
    """원본에만 있는 글과 PDF 에만 있는 글이 같은 자리에 있을 때, 서로 다른 글끼리 우연히 맞은 서너 글자('팻말이붙',
    '환한미소')에서 공백을 가져오지 않는다(닻 사이 틈은 거의 같은 글이거나 6글자 이상 맞은 덩어리만 받는다)."""
    same = [f"{k}번째 골목의 가게 앞에서 사람들이 {k * 7 + 3}번 버스를 기다린다." for k in range(8)]
    only_src = "이사장이 팻말이 붙은 사무실 문을 열고 들어서며 환한 미소를 짓는다."
    only_pdf = ["조직원들은창고앞에모여팻말이붙어있는문을본다", "누군가환한미소로답하자모두웃는다"]
    source = "\n".join(same[:4] + [only_src] + same[4:]) + "\n"
    pdf = [t.replace(" ", "") for t in same[:4]] + only_pdf + [t.replace(" ", "") for t in same[4:]]
    out, st = lines.respace(pdf, source)
    assert out[4:6] == only_pdf                                  # PDF 에만 있는 줄은 한 글자도 바뀌지 않는다
    assert out[:4] == same[:4] and out[6:] == same[4:]
    # 덩어리 길이 제한이 없으면 우연히 맞은 글자에서 원본의 공백이 옮겨 온다(제한이 이 경우를 막는다는 확인)
    monkeypatch.setattr(lines, "_DIFF_MIN", 1)
    out2, _ = lines.respace(pdf, source)
    assert out2[4:6] == ["조직원들은창고앞에모여팻말이 붙어있는문을본다", "누군가환한 미소로답하자모두웃는다"]


def test_respace_keeps_text_continuing_across_pdf_only_text():
    """쪽 머리말처럼 PDF 에만 있는 글 뒤에 이어지는 같은 글은 짧아도 맞춘다: 원본에서는 틈 머리, PDF 에서는 틈 끝에 붙은
    덩어리('창밖을바라', 인터뷰 fm54519: 원본에 군글자 '바라-보고' 가 있어 앞뒤 넓히기가 거기서 멈춘다)."""
    same = [f"{k}번째 골목의 가게 앞에서 사람들이 {k * 7 + 3}번 버스를 기다린다." for k in range(8)]
    source = "\n".join(same[:4] + ["창밖을 바라-보고 있는 검은 옷 차림의 영희."] + same[4:]) + "\n"
    pdf = [t.replace(" ", "") for t in same[:4]] + ["- 13 -", "영 화 인 터 뷰\t- 감 독 변 혁 -", "창밖을바라보고있는검은옷차림의영희."] \
        + [t.replace(" ", "") for t in same[4:]]
    out, _ = lines.respace(pdf, source)
    assert out[4:7] == ["- 13 -", "영 화 인 터 뷰\t- 감 독 변 혁 -", "창밖을 바라보고 있는 검은 옷 차림의 영희."]


def test_respace_does_not_match_gaps_longer_than_limit(monkeypatch):
    """닻(두 글에 한 번씩 나오는 8글자 조각) 사이 틈이 _GAP_MAX 보다 길면 그 틈은 맞추지 않고 줄 글 그대로 둔다."""
    same = [f"{k}번째 골목의 가게 앞에서 사람들이 {k * 7 + 3}번 버스를 기다린다." for k in range(8)]
    rep = ["그는 말이 없다. 그녀도 말이 없다."] * 12                 # 되풀이 글: 닻이 없다
    source = "\n".join(same[:4] + rep + same[4:]) + "\n"
    pdf = [t.replace(" ", "") for t in same[:4]] + ["- 7 -"] + [t.replace(" ", "") for t in rep] + ["- 8 -"] \
        + [t.replace(" ", "") for t in same[4:]]
    out, _ = lines.respace(pdf, source)
    assert out[5:17] == rep and out[4] == "- 7 -" and out[17] == "- 8 -"      # 틈이 짧으면 맞춘다(거의 같은 글)
    monkeypatch.setattr(lines, "_GAP_MAX", 100)
    out, _ = lines.respace(pdf, source)
    assert out[4:18] == pdf[4:18]                                              # 틈이 한도를 넘으면 그대로
    assert out[:4] == same[:4] and out[18:] == same[4:]


def _fake_lines(n: int, squeezed: set[int], loose: set[int] = frozenset()) -> tuple[list[dict], str]:
    """원본 문단 n 개(문단마다 낱말 8개 = 공백 7개)와, squeezed 차례의 줄은 띄어쓰기가 통째로 사라지고 loose 차례의 줄은
    공백 하나만 빠진 PDF 줄."""
    paras = [f"{k}번째 골목의 가게 앞에서 사람들이 {k * 7 + 3}번 버스를 기다린다." for k in range(n)]
    rows = [{"page": 1 + k // 40, "x": 0.1667, "y": round((k % 40) / 50, 4),
             "text": p.replace(" ", "") if k in squeezed else p.replace(" ", "", 1) if k in loose else p,
             "bold": False, "src": "hwp", "conf": None}
            for k, p in enumerate(paras)]
    return rows, "\n".join(paras) + "\n"


def test_restore_spacing_only_above_threshold():
    """공백 손실이 문턱(RESPACE_LOSS)을 넘는 파일만 줄 글을 바꾼다. 문턱 아래 파일은 띄어쓰기가 통째로 사라진 줄이 몇 줄
    있어도 한 글자도 바꾸지 않고 기록도 없다(그런 줄과 이름 뒤 탭 손실은 따로 정할 일: notes/parser.md 알려진 문제)."""
    rows, source = _fake_lines(60, squeezed=set(), loose={5, 17, 40})          # 공백 3개만 빠짐
    before = [dict(r) for r in rows]
    assert lines.restore_spacing(rows, source) is None and rows == before
    rows, source = _fake_lines(60, squeezed={5, 17, 40}, loose={8})            # 3/60 줄이 통째로: 손실 0.052
    before = [dict(r) for r in rows]
    assert lines.restore_spacing(rows, source) is None and rows == before
    assert lines.respace([r["text"] for r in rows], source)[1]["loss"] == 0.0524
    rows, source = _fake_lines(60, squeezed=set(range(0, 60, 2)))              # 절반: 손실 0.5
    before = [dict(r) for r in rows]
    info = lines.restore_spacing(rows, source)
    assert info == {"from": "source", "loss": 0.5, "matched": 1.0, "lines": 30, "spaces": 210, "tabs": 0,
                    "removed": 0, "lowered": 0}
    assert [r["text"] for r in rows] == source.splitlines()
    assert [(r["x"], r["y"], r["page"]) for r in rows] == [(r["x"], r["y"], r["page"]) for r in before]
    # 원본 글이 비었거나 견줄 경계가 너무 적으면 바꾸지 않고 까닭을 돌려준다
    few, src = _fake_lines(10, squeezed=set(range(10)))
    keep = [dict(r) for r in few]
    assert lines.restore_spacing(few, src) == {"from": None, "skipped": "few_bounds:70"} and few == keep
    assert lines.restore_spacing(rows, None) == {"from": None, "skipped": "source_empty"}


def _hancom_doc(tmp_path, monkeypatch, doc_id: str, names: list[str], rows_of: dict[str, list[dict]]):
    """임시 폴더에 가짜 원본 파일을 두고, 한컴 변환 PDF 에서 줄을 만드는 문서처럼 꾸민다(PDF 읽기만 갈아 끼움)."""
    from sinkprep import paths
    for name in ("DOCS", "STATE", "PROBLEMS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    files = []
    for nm in names:
        f = tmp_path / nm
        if not f.exists():
            f.write_bytes(nm.encode("utf-8"))
        pdf = tmp_path / (nm + ".pdf")
        pdf.write_bytes(b"%PDF-" + nm.encode("utf-8"))
        files.append(str(f))
    monkeypatch.setattr(lines, "input_for", lambda doc, k, f, fmt: ("pdf", Path(f + ".pdf"), "hwp"))
    monkeypatch.setattr(lines, "lines_from_pdf", lambda p, src: [dict(r) for r in rows_of[Path(p).name[:-4]]])
    return {"doc_id": doc_id, "files_list": files, "file_formats_list": ["hwp5"] * len(files)}


def test_build_doc_marks_restored_file_in_header(tmp_path, monkeypatch):
    """한컴 변환 PDF 경로 파일 가운데 공백 손실이 문턱을 넘는 파일만 되살리고, 되살린 파일은 머리 행 made.spacing 에
    파일 차례(file)와 함께 표시한다(검사 도구도 통과). 문턱 아래 파일은 줄 글 그대로이고 표시도 없다.
    원본 글은 convert.fallback_text 로 읽고, 원본 파일이 바뀌면(지문 sources) 다시 만든다."""
    from sinkprep import convert, paths, validate
    rows, source = _fake_lines(60, squeezed=set(range(0, 60, 2)))
    rows2, source2 = _fake_lines(60, squeezed={3, 4, 5})
    rows3, source3 = _fake_lines(60, squeezed=set(range(0, 60, 3)))
    names = ["a [9100001].hwp", "b [9100001].hwp", "c [9100001].hwp"]
    doc = _hancom_doc(tmp_path, monkeypatch, "fm9100001", names, dict(zip(names, [rows, rows2, rows3])))
    texts = dict(zip(names, [source, source2, source3]))
    calls = []

    def fake_read(path, fmt, notes=None):
        calls.append((Path(path).name, fmt))
        return texts[Path(path).name], "fallback_fake"

    monkeypatch.setattr(convert, "fallback_text", fake_read)
    assert lines.build_doc(doc) == ("ok", 180)
    assert calls == [(nm, "hwp5") for nm in names]
    out = paths.DOCS / "fm9100001" / "lines.jsonl"
    got = util.read_jsonl(out)
    assert got[0]["made"]["version"] == lines.VERSION
    assert got[0]["made"]["spacing"] == [
        {"file": 0, "from": "source", "loss": 0.5, "matched": 1.0, "lines": 30, "spaces": 210, "tabs": 0,
         "removed": 0, "lowered": 0},
        {"file": 2, "from": "source", "loss": 0.3333, "matched": 1.0, "lines": 20, "spaces": 140, "tabs": 0,
         "removed": 0, "lowered": 0}]
    assert [r["text"] for r in got[1:]] == source.splitlines() + [r["text"] for r in rows2] + source3.splitlines()
    assert [r["n"] for r in got[1:]] == list(range(180))
    assert [r["file"] for r in got[1:]] == [0] * 60 + [1] * 60 + [2] * 60
    assert validate.validate_lines(out) == []
    assert lines.build_doc(doc) == ("skip", -1)                     # 지문이 같으면 건너뜀
    Path(doc["files_list"][1]).write_bytes(b"changed")              # 원본 파일이 바뀌면 다시 만든다
    assert lines.build_doc(doc) == ("ok", 180)
    # 글자층 PDF 원본(pdf_text)은 원본 문서 글이 따로 없으므로 손대지 않고 표시도 없다
    monkeypatch.setattr(lines, "input_for", lambda doc, k, f, fmt: ("pdf", Path(tmp_path / "a [9100001].hwp.pdf"), "pdf_text"))
    monkeypatch.setattr(lines, "lines_from_pdf", lambda p, src: [dict(r) for r in rows])
    doc2 = {"doc_id": "fm9100002", "files_list": ["b [9100002].pdf"], "file_formats_list": ["pdf_text"]}
    lines.build_doc(doc2)
    got2 = util.read_jsonl(paths.DOCS / "fm9100002" / "lines.jsonl")
    assert "spacing" not in got2[0]["made"] and [r["text"] for r in got2[1:]] == [r["text"] for r in rows]


def test_build_doc_records_unreadable_source_and_retries(tmp_path, monkeypatch):
    """원본 글을 못 읽으면 줄은 PDF 에서 만든 그대로 두고 머리 행에 까닭을 남긴다. 한때의 실패(시간 초과)면 지문을 맞지
    않게 남겨 다음 실행이 다시 만들고, 읽는 코드가 못 읽는 파일(늘 같은 실패)이면 건너뛴다."""
    from sinkprep import convert, paths, validate
    rows, source = _fake_lines(60, squeezed=set(range(0, 60, 2)))
    doc = _hancom_doc(tmp_path, monkeypatch, "fm9100003", ["c [9100003].doc"], {"c [9100003].doc": rows})
    doc["file_formats_list"] = ["doc"]
    out = paths.DOCS / "fm9100003" / "lines.jsonl"
    state = {"fail": TimeoutError("antiword 120s")}

    def fake_read(path, fmt, notes=None):
        if state["fail"] is not None:
            raise state["fail"]
        return source, "fallback_fake"

    monkeypatch.setattr(convert, "fallback_text", fake_read)
    assert lines.build_doc(doc) == ("ok", 60)
    got = util.read_jsonl(out)
    assert got[0]["made"]["spacing"] == [{"file": 0, "from": None, "skipped": "source_unreadable",
                                          "error": "TimeoutError: antiword 120s"}]
    assert [r["text"] for r in got[1:]] == [r["text"] for r in rows] and validate.validate_lines(out) == []
    state["fail"] = None                                            # 다음 실행: 읽기가 되면 다시 만들어 되살린다
    assert lines.build_doc(doc) == ("ok", 60)
    got = util.read_jsonl(out)
    assert got[0]["made"]["spacing"][0]["from"] == "source" and [r["text"] for r in got[1:]] == source.splitlines()
    assert lines.build_doc(doc) == ("skip", -1)
    # 늘 같은 실패(읽는 코드가 못 읽는 파일): 까닭만 남기고 다음 실행은 건너뛴다. 경로가 든 예외 글 뒷부분은 적지 않는다
    state["fail"] = ValueError(r"antiword_rc1:C:\somewhere\c.doc is not a Word Document.")
    assert lines.build_doc(doc, force=True) == ("ok", 60)
    got = util.read_jsonl(out)
    assert got[0]["made"]["spacing"] == [{"file": 0, "from": None, "skipped": "source_unreadable",
                                          "error": "ValueError: antiword_rc1"}]
    assert lines.build_doc(doc) == ("skip", -1)
    # 한때의 실패는 --force 로 만든 뒤에도 다음 실행이 다시 한다
    state["fail"] = TimeoutError("again")
    assert lines.build_doc(doc, force=True) == ("ok", 60)
    state["fail"] = None
    assert lines.build_doc(doc) == ("ok", 60)
    # 되살리기가 예외로 끝나도 줄은 PDF 에서 만든 그대로 쓰고 까닭을 남긴다
    def boom(*a, **k):
        raise RuntimeError("bad align")

    monkeypatch.setattr(lines, "respace", boom)
    assert lines.build_doc(doc, force=True) == ("ok", 60)
    got = util.read_jsonl(out)
    assert got[0]["made"]["spacing"] == [{"file": 0, "from": None, "skipped": "error", "error": "RuntimeError: bad align"}]
    assert [r["text"] for r in got[1:]] == [r["text"] for r in rows]
    # 원본 파일이 없으면(옮겨졌거나 지워짐) 견주지 못한 까닭만 남긴다
    Path(doc["files_list"][0]).unlink()
    assert lines.build_doc(doc) == ("ok", 60)
    got = util.read_jsonl(out)
    assert got[0]["made"]["spacing"] == [{"file": 0, "from": None, "skipped": "source_missing"}]
    assert lines.build_doc(doc) == ("skip", -1)
