"""TYPES.md 와 types.csv 생성: 코드가 원본 파일을 어떤 유형으로 나누는지(기준·값·문서 수)와 문서별 유형 표.

사용: .venv/Scripts/python -I tools/make_types.py [--rules] [--out-dir DIR]
  --rules        LLM 판으로 바뀐 문서도 보관된 규칙 판(cache/parse_rules)으로 센다(규칙 파서만의 결과를 볼 때).
  --out-dir DIR  TYPES.md 와 types.csv 를 DIR 에 쓴다(기본 저장소 맨 위).
숫자는 모두 지금 디스크의 결과물에서 다시 센다(손으로 쓴 숫자 없음):
catalog.csv, cache/convert_log.jsonl, cache/ocr, docs/*/lines.jsonl(머리 행의 made.spacing 포함), docs/*/script.json,
qa/needs_review.csv.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import catalog, paths, util  # noqa: E402

CAT_KO = {"ko_film": "한국영화", "foreign_film": "외국영화"}

# ---------------------------------------------------------------- 1. 파일 형식 (detect.real_format 의 판별 규칙)
FORMATS = [
    ("hwp5", "OLE 복합 문서이고 FileHeader 스트림이 `HWP Document File` 로 시작"),
    ("hwp3", "파일 머리가 `HWP Document File V3`"),
    ("hwpx", "ZIP 이고 mimetype 이 `application/hwp+zip` 이거나 Contents/section*.xml 이 있음"),
    ("doc", "OLE 복합 문서이고 WordDocument 스트림이 있음"),
    ("docx", "ZIP 이고 word/document.xml 이 있음"),
    ("rtf", "`{\\rtf` 로 시작"),
    ("pdf_text", "`%PDF-` 이고 글자 50자 미만인 쪽이 절반보다 적음"),
    ("pdf_scan", "`%PDF-` 이고 글자 50자 미만인 쪽이 절반 이상(그림만 있는 스캔)"),
    ("pdf_broken", "`%PDF-` 인데 열리지 않거나 암호가 걸렸거나 0쪽"),
    ("txt", "UTF-8·UTF-16·cp949 로 풀리고 제어 문자가 거의 없음"),
    ("xls", "OLE 의 Workbook/Book 스트림, 또는 ZIP 의 xl/workbook.xml"),
    ("archive", "zip(`PK`)·rar(`Rar!`)·alz(`ALZ`) 압축: 풀어서 안의 파일을 다시 판별"),
    ("other", "실행 파일(`MZ`)·그림·알 수 없는 바이너리"),
]
EXT_OF = {".hwp": {"hwp3", "hwp5"}, ".doc": {"doc"}, ".docx": {"docx"}, ".pdf": {"pdf_text", "pdf_scan", "pdf_broken"},
          ".txt": {"txt"}, ".rtf": {"rtf"}, ".xls": {"xls"}}

# ---------------------------------------------------------------- 3. 글자를 얻는 경로
ROUTES = [
    ("hancom_pdf", "한컴 변환 PDF 의 글자층", "한글 2024 로 PDF 를 만들고 글자 좌표로 줄을 만든다(레이아웃 있음)",
     "convert.convert_file, lines.lines_from_pdf"),
    ("fallback_text", "대체 경로 평문", "한컴이 못 열거나 PDF 에서 글자가 3% 넘게 빠진 파일. 파일을 직접 풀어 글만 얻는다(레이아웃 없음)",
     "convert.fallback_text, lines.lines_from_text"),
    ("pdf_textlayer", "원본 PDF 의 글자층", "글자 좌표로 줄을 만든다(레이아웃 있음)", "lines.lines_from_pdf"),
    ("ocr", "OCR", "스캔 PDF 와 글자층이 깨진 PDF 를 300dpi 그림으로 읽는다(RapidOCR)", "ocr.ocr_pdf, lines.lines_from_ocr"),
    ("plain_text", "평문", "인코딩만 풀어 줄로 나눈다(레이아웃 없음)", "lines.lines_from_text"),
    ("xls_table", "엑셀 표", "한 행을 한 줄로, 칸 사이는 탭", "lines.lines_from_xls"),
    ("none", "줄 없음", "읽은 줄이 없음", ""),
]
ROUTE_KO = {k: ko for k, ko, _, _ in ROUTES}
WORDPROC_SRC = {"hwp", "doc", "docx", "rtf"}

# ---------------------------------------------------------------- 5. 대사 표기
FORMS = [
    ("own_line", "이름 줄", "이름만 한 줄에 적고 대사는 다음 줄부터", "`철수` 다음 줄 `어디 가?`",
     "DocParser._vocab (block_mode·flat_mode), _collapsed_cues"),
    ("tab", "이름⇥대사", "이름과 대사 사이가 탭(칸 띄움)", "`철수⇥어디 가?`", "DocParser._inline_split_raw (tab)"),
    ("colon", "이름 : 대사", "이름 뒤 쌍점(쌍반점·빗금도 같은 표기로 봄)", "`철수 : 어디 가?`, `철수/ 어디 가?`",
     "DocParser._inline_split_raw (colon)"),
    ("space1", "이름 대사", "이름 뒤 공백 하나뿐", "`철수 어디 가?`", "DocParser._vocab (space_vocab: 줄 첫 낱말 빈도)"),
    ("space2", "이름  대사", "이름 뒤 공백 둘 이상", "`철수   어디 가?`", "DocParser._inline_split_raw (space2)"),
    ("quote", "이름\"대사\"", "이름 뒤(공백 없이 또는 하나 두고) 따옴표", "`철수\"어디 가?\"`", "DocParser._inline_split_raw (quote)"),
    ("bracket", "[이름] 대사", "이름을 대괄호로 쌈", "`[철수] 어디 가?`", "DocParser._bracketed"),
    ("angle", "<이름> 대사", "이름을 꺾쇠로 쌈(대사가 다음 줄에 와도)", "`<철수> 어디 가?`", "DocParser._inline_split_raw (angle)"),
    ("joined", "이름대사", "구분 없이 붙음", "`철수어디 가?`", ""),
    ("other", "기타", "요소 첫머리에서 이름 부분을 찾지 못함", "", ""),
    ("none", "대사 없음", "이름이 적힌 대사가 5개 미만(시놉시스, 자막 받아쓰기 등)", "", ""),
]
FORM_KO = {k: ko for k, ko, _, _, _ in FORMS}
MIN_DIALOGUES = 5         # 이름 적힌 대사가 이보다 적으면 대표 표기를 정하지 않는다(none)
DECO_MIN_SHARE = 0.3      # 이름 꾸밈(띄운 이름·글머리표)이 이 비율 이상이면 문서의 이름 꾸밈으로 적는다
DECO_KO = {"letterspaced": "글자 사이를 띄운 이름(`철 수`)", "bullet": "글머리표가 붙은 이름(`*철수`)"}

_OPEN = {"[": "]", "【": "】", "<": ">", "〈": "〉", "《": "》"}
_BULLETS = "*-●○◎■□▶▷·•※◆◇"
_QUOTES = ('"', "“", "‘", "''")      # 파서 _RX_QUOTE 가 받는 여는 따옴표
_COLONS = ":："
_WS = " \t"
_LSP = re.compile(r"[가-힣](?:[ \t]+[가-힣]){1,3}")


def _core(speaker_raw: str) -> str:
    """speaker_raw 에서 이름을 싼 괄호(<이름>, [이름])와 꼬리 괄호((V.O.), (계속), 지시)를 뗀 이름 부분."""
    s = speaker_raw.strip()
    if len(s) > 2 and _OPEN.get(s[0]) == s[-1]:
        s = s[1:-1].strip()
    i = s.find("(")
    if i > 0:
        s = s[:i]
    return s.strip().rstrip(":：;").strip()


def _match(raw: str, core: str) -> tuple[int, str] | None:
    """raw 첫머리에서 이름(core)을 공백을 건너뛰며 맞춘다 -> (이름 끝 위치, 꾸밈) 또는 None."""
    key = [c for c in core if not c.isspace()]
    if not key:
        return None
    n, i, deco, close = len(raw), 0, "", ""
    while i < n and raw[i] in _WS:
        i += 1
    if i < n and raw[i] != key[0]:
        if raw[i] in _OPEN:
            deco, close = ("bracket" if raw[i] in "[【" else "angle"), _OPEN[raw[i]]
            i += 1
        elif raw[i] in _BULLETS:
            deco = "bullet"
            i += 1
        while i < n and raw[i] in _WS:
            i += 1
    start, j = i, 0
    while i < n and j < len(key):
        if raw[i] == key[j]:
            j += 1
            i += 1
        elif raw[i] in _WS:
            i += 1
        else:
            return None
    if j < len(key):
        return None
    if not deco and _LSP.fullmatch(raw[start:i]):
        deco = "letterspaced"
    if close:
        k = i
        while k < n and raw[k] in _WS:
            k += 1
        if k < n and raw[k] == close:
            i = k + 1
    return i, deco


def _sep(rest: str) -> str:
    """이름 바로 뒤 글(rest)로 표기를 정한다. 차례는 파서(DocParser._inline_split_raw)와 같다:
    탭, 쌍점(쌍반점·빗금), 공백 둘 이상, 따옴표, 공백 하나."""
    line = rest.split("\n", 1)[0]
    i, gap = 0, []                  # gap: 이름과 대사 사이의 공백·구분 글자(한 줄 안에서 닫히는 꼬리 괄호는 건너뜀)
    while i < len(line):
        c = line[i]
        if c in _WS or c in _COLONS:
            gap.append(c)
        elif c in ";/" and line[i + 1:i + 2] in (" ", "\t"):
            gap.append(":")         # '이름; 대사', '이름/ 대사' 는 파서가 쌍점 표기로 받는다(_RX_SEMI_SLASH)
        elif c == "(" and ")" in line[i:]:
            i = line.index(")", i)
            gap.append("|")         # 괄호 앞뒤 공백을 한 덩어리로 세지 않게 끊는다
        else:
            break
        i += 1
    zone, body = "".join(gap), line[i:]
    if "\t" in zone:
        return "tab"
    if any(c in zone for c in _COLONS):
        return "colon"
    if not body:
        return "own_line"
    if "  " in zone:
        return "space2"
    if body.startswith(_QUOTES):
        return "quote"
    return "space1" if " " in zone else "joined"


def dialogue_form(el: dict) -> tuple[str, str] | None:
    """대사 요소 하나의 (표기, 이름 꾸밈). 이름이 적히지 않았거나 앞 대사에서 이어받은 화자면 None."""
    sp = el.get("speaker_raw")
    if not sp or el.get("speaker_inferred"):
        return None
    raw = el.get("raw") or ""
    m = _match(raw, _core(sp))
    if m is None:
        return "other", ""
    end, deco = m
    if deco in ("bracket", "angle"):
        return deco, ""
    return _sep(raw[end:]), deco


# ---------------------------------------------------------------- 파서가 켠 특수 분기 (script.json quality.notes)
BRANCHES = [
    ("flat_names", r"^대사 표기: 들여쓰기 없는 이름 줄", "들여쓰기 없는 이름 줄(빈 줄 뒤 이름, 빈 줄 앞까지 대사)", "DocParser._vocab flat_mode"),
    ("space1_vocab", r"^대사 표기: 이름 뒤 공백 하나", "이름 뒤 공백 하나: 줄 첫 낱말 빈도로 이름을 찾음", "DocParser._vocab space_vocab"),
    ("letterspaced_names", r"^대사 표기: 글자 사이를 띄운 이름", "글자 사이를 띄운 이름", "DocParser._letterspaced"),
    ("collapsed_cues", r"^대사 표기: 대사가 지문 여백에 놓인 대문자 이름 줄", "대사가 지문 여백에 놓인 대문자 이름 줄(영어 대본)",
     "DocParser._collapsed_cues"),
    ("bullet_names", r"^대사 표기: 글머리표", "글머리표가 붙은 이름", "DocParser._candidates"),
    ("bracket_names", r"^대사 표기: 대괄호 이름", "대괄호 이름", "DocParser._bracketed"),
    ("voice_cue", r"^대사 표기: '소리/목소리 : 대사'", "'소리 : 대사'(화자 없는 화면 밖 소리)", "DocParser._candidates"),
    ("group_cue", r"^대사 표기: '우리 : 대사'", "'우리 : 대사'(무리 화자)", "DocParser._candidates"),
    ("shifted_pages", r"^대사 표기: .*쪽 밀림 맞춤", "쪽마다 여백이 달라 쪽 밀림을 맞춤", "parse_layout.analyze_x"),
    ("scene_indented_series", r"^씬 머리: 들여쓰거나 가운데 놓인 번호 씬 머리", "들여쓰거나 가운데 놓인 번호 씬 머리", "DocParser._accept_headings"),
    ("scene_upper_slug", r"^씬 머리: INT\./EXT\.·번호가 없어", "INT./EXT.·번호 없이 대문자 짧은 줄이 씬 머리", "DocParser._accept_headings"),
    ("scene_restart", r"^씬 번호가 \d+번 줄에서 1로 다시 시작", "씬 번호가 1로 다시 시작(앞 번호는 인물표·설정)", "DocParser._accept_headings"),
    ("front_dialogue", r"^앞머리: 첫 씬 머리 전에 대사가 있어", "첫 씬 머리 앞에 대사가 있음", "DocParser._front_end"),
    ("front_none", r"^앞머리: 없음", "앞머리 없이 첫 줄부터 본문", "DocParser._front_end"),
    ("speaker_inherit", r"^화자 이어받기", "이름 줄 없이 이어진 대사는 앞 화자를 이어받음(speaker_inferred)", "DocParser._build"),
    ("vo_resolved", r"^\(V\.O\.\)", "(V.O.) 를 화면 밖 소리와 내레이션으로 가름", "DocParser._resolve_vo"),
    ("n_ambiguous", r"^\(N\)", "(N) 을 밤과 내레이션으로 가름", "DocParser._dialogue_fields"),
    ("subtitle_file", r"^자막 파일 꼴", "자막 파일 꼴(시각 줄·자막 번호를 뺌)", "DocParser.parse"),
    # parse 0.3.x 에서 더한 분기
    ("scene_colon_heads", r"^씬 머리: 쌍점이 든 번호 씬 머리",
     "쌍점이 든 번호 씬 머리(`12. 장소 : 설명 / 밤`)를 받아들인 씬 머리와 같은 칸에서 빈 번호를 차례대로 메울 때 받음",
     "DocParser._colon_heads"),
    ("paren_names", r"^대사 표기: 괄호에 든 이름", "괄호에 든 이름 `(이름)⇥대사`: 이름표에 있는 인물의 화면 밖 목소리(off)",
     "DocParser._paren_names"),
    ("wordless_turns", r"^대사 표기: 말 없이 점·괄호만 있는 한 줄 차례",
     "말 없이 점·괄호만 있는 한 줄 차례(`이름⇥.......`)를 이름 줄이 아닌 한 줄 대사로 봄", "DocParser._vocab, _wordless"),
    ("tight_cont", r"^대사 이어짐: 지문 앞에 빈 줄을 두는 문서",
     "지문 앞에 늘 빈 줄을 두는 문서: 대사 바로 아래 붙여 쓴 들여쓴 줄은 앞 대사의 이어짐", "DocParser._cont (tight_cont)"),
]
_BRANCH_RE = [(k, re.compile(p)) for k, p, _, _ in BRANCHES]
PARSE_03_BRANCHES = ("scene_colon_heads", "paren_names", "wordless_turns", "tight_cont")   # parse 0.3.x 의 새 꼬리표

CSV_FIELDS = ["doc_id", "category", "title", "real_format", "files", "route", "convert_method", "ocr_why", "two_up",
              "script_lang", "is_translation", "version_label", "content", "is_primary", "parser", "lines", "scenes",
              "dialogues", "dialogue_form", "form_share", "name_deco", "parser_branches", "score", "needs_review",
              "spacing"]           # 새 칸은 끝에만 더한다(앞 칸의 차례·뜻은 그대로)
SPACING_KO = {"source": "한컴 PDF 줄의 공백·탭을 원본 문서 글로 맞춤", "skipped": "원본 글을 읽지 못해 공백 손실을 견주지 못함"}


def read_lines_stat(doc_id: str) -> tuple[dict[int, Counter], dict[int, int], int]:
    """lines.jsonl -> (파일별 src 줄 수, 파일별 가장 큰 쪽 번호, 줄 수)."""
    src: dict[int, Counter] = defaultdict(Counter)
    maxpage: dict[int, int] = defaultdict(int)
    n = 0
    p = paths.DOCS / doc_id / "lines.jsonl"
    if not p.exists():
        return src, maxpage, 0
    with open(p, encoding="utf-8") as f:
        f.readline()                                    # 머리 행
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            k = x.get("file") or 0
            src[k][x["src"]] += 1
            if x.get("page"):
                maxpage[k] = max(maxpage[k], x["page"])
            n += 1
    return src, maxpage, n


def lines_head(doc_id: str) -> dict:
    """lines.jsonl 머리 행(첫 줄). 없거나 머리 행이 아니면 빈 dict."""
    p = paths.DOCS / doc_id / "lines.jsonl"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        first = f.readline()
    try:
        head = json.loads(first) if first.strip() else {}
    except ValueError:
        return {}
    return head if isinstance(head, dict) and "schema" in head else {}


def spacing_of(head: dict) -> str:
    """머리 행 made.spacing -> 공백 복원 값: source(원본 글로 맞춤), skipped(견주지 못함), 파일마다 다르면 + 로 이음.
    기록이 없으면 빈 값(견줬고 문턱 아래, 한컴 PDF 경로가 아님, 또는 공백 손실을 재기 전의 lines 판)."""
    recs = ((head.get("made") or {}).get("spacing") or []) if isinstance(head.get("made"), dict) else []
    vals = ["source" if r.get("from") == "source" else "skipped" for r in recs if isinstance(r, dict)]
    return "+".join(dict.fromkeys(vals))


def pdf_pages(pdf: Path) -> int | None:
    import pymupdf
    try:
        with pymupdf.open(pdf) as d:
            return d.page_count
    except Exception:  # noqa: BLE001 - 열리지 않는 PDF 는 펼침 판단에서 뺀다
        return None


def doc_types(r: dict, log: dict, review: dict[str, list[str]], rules_only: bool = False) -> dict:
    doc_id = r["doc_id"]
    src, maxpage, n_lines = read_lines_stat(doc_id)
    routes, methods, whys, two_up = [], [], [], False
    for k, (rel, fmt) in enumerate(zip(r["files_list"], r["file_formats_list"])):
        s = src[k].most_common(1)[0][0] if src.get(k) else None
        row = log.get((doc_id, k))
        if row:
            methods.append(row["method"])
        if s == "ocr":
            route = "ocr"
            oj = paths.OCR_CACHE / doc_id / f"{k}.json"
            if oj.exists():
                o = util.read_json(oj)
                whys.append(o.get("why") or "?")
                two_up = two_up or any(pg.get("parts") for pg in o.get("pages", []))
        elif s == "pdf_text":
            route = "pdf_textlayer"
            pc = pdf_pages(paths.absp(rel))
            two_up = two_up or (pc is not None and maxpage[k] > pc)
        elif s in WORDPROC_SRC:
            pdf = paths.PDF_CACHE / doc_id / f"{k}.pdf"
            if pdf.exists():
                route = "hancom_pdf"
                pc = pdf_pages(pdf)
                two_up = two_up or (pc is not None and maxpage[k] > pc)
            else:
                route = "fallback_text"
        elif s == "txt":
            route = "plain_text"
        elif s == "xls":
            route = "xls_table"
        else:
            route = "none"
        routes.append(route)

    sp = paths.DOCS / doc_id / "script.json"
    parser, scenes, n_dlg, score, form, share, deco, branches = "", "", 0, "", "none", "", "", []
    llm_now = False
    if sp.exists():
        s = rules = util.read_json(sp)
        if s.get("parser", {}).get("name") != "sink-parse":     # LLM 판: 규칙 판은 cache/parse_rules 에 보관돼 있다
            llm_now = True
            rp = paths.CACHE / "parse_rules" / f"{doc_id}.json"
            rules = util.read_json(rp) if rp.exists() else None
            if rules_only and rules is not None:
                s = rules
        parser = s.get("parser", {}).get("name", "")
        scenes = len(s.get("scenes") or [])
        score = (s.get("quality") or {}).get("score", "")
        forms, decos = Counter(), Counter()
        for el in s.get("elements", []):
            if el.get("type") != "dialogue":
                continue
            n_dlg += 1
            fd = dialogue_form(el)
            if fd:
                forms[fd[0]] += 1
                decos[fd[1]] += 1
        named = sum(forms.values())
        if named >= MIN_DIALOGUES:
            form, top = forms.most_common(1)[0]
            share = f"{top / named:.2f}"
            dk, dn = max(((k, v) for k, v in decos.items() if k), key=lambda kv: kv[1], default=("", 0))
            if dn / named >= DECO_MIN_SHARE:
                deco = dk
        # 특수 분기는 규칙 파서가 적은 notes 에서 읽는다(LLM 판 문서는 보관된 규칙 판에서)
        notes = ((rules or s).get("quality") or {}).get("notes") or []
        for key, rx in _BRANCH_RE:
            if any(rx.search(nt) for nt in notes):
                branches.append(key)

    note = r.get("note") or ""
    content = "|".join(x for x in note.split(";") if x.startswith(("content:", "tv_")))
    head = lines_head(doc_id)
    made = head.get("made") if isinstance(head.get("made"), dict) else {}
    loss = [rec.get("loss") for rec in made.get("spacing") or [] if isinstance(rec, dict) and rec.get("from") == "source"]
    return {
        "doc_id": doc_id, "category": r["category"], "title": r.get("title_norm") or r.get("title_raw") or "",
        "real_format": r["real_format"], "files": len(r["files_list"]),
        "route": "+".join(dict.fromkeys(routes)), "convert_method": "+".join(dict.fromkeys(methods)),
        "ocr_why": "+".join(dict.fromkeys(whys)), "two_up": "true" if two_up else "false",
        "script_lang": r["script_lang"], "is_translation": r["is_translation"], "version_label": r["version_label"],
        "content": content, "is_primary": r["is_primary"], "parser": parser, "lines": n_lines, "scenes": scenes,
        "dialogues": n_dlg, "dialogue_form": form, "form_share": share, "name_deco": deco,
        "parser_branches": "|".join(branches), "score": score, "needs_review": "|".join(review.get(doc_id, [])),
        "spacing": spacing_of(head),
        "_llm_now": llm_now,        # docs 의 script.json 이 지금 LLM 판인지(types.csv 에는 쓰지 않음)
        "_lines_version": str(made.get("version") or ""),     # lines.jsonl 을 만든 lines 판(types.csv 에는 쓰지 않음)
        "_spacing_loss": loss,      # 공백 복원한 파일의 공백 손실(types.csv 에는 쓰지 않음)
    }


def table(head: list[str], body: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in body:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def counts(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="TYPES.md 와 types.csv 생성")
    ap.add_argument("--rules", action="store_true", help="LLM 판 문서도 보관된 규칙 판으로 센다")
    ap.add_argument("--out-dir", default=None, help="TYPES.md 와 types.csv 를 쓸 폴더(기본 저장소 맨 위)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir) if a.out_dir else ROOT
    rows = catalog.load()
    docs = [r for r in rows if r["status"] != "excluded"]
    excl = [r for r in rows if r["status"] == "excluded"]
    log = {}
    lp = paths.CACHE / "convert_log.jsonl"
    if lp.exists():
        for x in util.read_jsonl(lp):
            log[(x["doc_id"], x["k"])] = x
    review: dict[str, list[str]] = defaultdict(list)
    if paths.NEEDS_REVIEW.exists():
        with open(paths.NEEDS_REVIEW, encoding="utf-8-sig", newline="") as f:
            for x in csv.DictReader(f):
                if x["doc_id"] and x["problem"] not in review[x["doc_id"]]:
                    review[x["doc_id"]].append(x["problem"])

    rules_only = a.rules
    T = [doc_types(r, log, review, rules_only) for r in docs]
    cat_of = {r["doc_id"]: r["category"] for r in docs}

    def split(ids) -> tuple[int, int, int]:
        ids = list(ids)
        return (len(ids), sum(cat_of[i] == "ko_film" for i in ids), sum(cat_of[i] == "foreign_film" for i in ids))

    L: list[str] = []
    w = L.append
    w("# 원본 파일 유형 분류 (코드가 나누는 기준과 결과)")
    w("")
    w(f"만든 시각: {time.strftime('%Y-%m-%d %H:%M')} · 생성기 `tools/make_types.py`(모든 숫자는 디스크 결과물에서 다시 셈) · 문서별 표는 `types.csv`")
    if rules_only:
        w("")
        w("규칙 파서 판 기준(`--rules`): LLM 판으로 바뀐 문서도 보관된 규칙 판(cache/parse_rules)으로 센다.")
    w("")
    raw_files = [p for p in paths.RAW.rglob("*") if p.is_file() and p.name != "manifest.csv"]
    w(f"원본 `filmmakers_scripts/` 파일 {len(raw_files)}개를 코드는 다섯 번 나눈다. 앞 단계의 유형이 뒤 단계의 처리 경로를 정한다.")
    w("")
    w("| 차례 | 나누는 기준 | 코드 | 결과가 남는 곳 |")
    w("|---|---|---|---|")
    w("| 1 | 파일 형식(앞 바이트·내부 구조) | `sinkprep/detect.py` | catalog.csv `real_format`, `file_formats` |")
    w("| 2 | 문서로 묶기와 제외 | `sinkprep/catalog.py` | catalog.csv 한 행 = 한 문서, `status`, `note` |")
    w("| 3 | 글자를 얻는 경로 | `sinkprep/convert.py`, `ocr.py`, `lines.py` | cache/convert_log.jsonl, cache/ocr, lines.jsonl `src` |")
    w("| 4 | 내용 종류·언어·판본 | `sinkprep/catalog.py`, `review.py` | catalog.csv `note`, `script_lang`, `version_label` |")
    w("| 5 | 대본 꼴(대사 표기·씬 머리) | `sinkprep/parse.py`, `parse_layout.py`, `parse_text.py` | script.json 요소, `quality.notes` |")
    w("")

    # ---------- 1
    w("## 1. 파일 형식")
    w("")
    w("`detect.real_format()` 이 확장자가 아니라 앞 바이트와 내부 구조로 정한다. 압축은 풀어서 안의 파일을 다시 판별한다.")
    w("")
    fmt_files = Counter(f for r in rows for f in r["file_formats_list"])
    fmt_docs = Counter(r["real_format"] for r in docs)
    body = []
    unlisted = sorted((set(fmt_files) | set(fmt_docs)) - {f for f, _ in FORMATS})      # 표에 없는 값(예: mixed)
    for fmt, rule in FORMATS + [(f, "") for f in unlisted]:
        n, ko, fo = split(r["doc_id"] for r in docs if r["real_format"] == fmt)
        body.append([f"`{fmt}`", rule, fmt_files.get(fmt, 0), n, ko, fo])
    L.extend(table(["형식", "판별 규칙", "파일", "문서", "한국영화", "외국영화"], body))
    w("")
    w(f"- 파일 {sum(fmt_files.values())}개(원본 {len(raw_files)}개 + 압축에서 나온 파일), 문서 {len(docs)}개"
      f"(한국영화 {sum(r['category'] == 'ko_film' for r in docs)}, 외국영화 {sum(r['category'] == 'foreign_film' for r in docs)})")
    disguised = []
    for r in rows:
        for f, fmt in zip(r["files_list"], r["file_formats_list"]):
            e = Path(f).suffix.lower()
            if e in EXT_OF and fmt not in EXT_OF[e] and fmt != "archive":
                disguised.append(f"{e} 인데 실제는 {fmt}")
    w(f"- 확장자와 실제 형식이 다른 파일 {len(disguised)}개: " + counts(Counter(disguised)))
    w("")

    # ---------- 2
    w("## 2. 문서로 묶기와 제외")
    w("")
    w("`catalog.group_post()` 가 게시글 하나의 파일들을 문서로 묶는다. 같은 바탕 이름에 번호가 이어지는 파일(상·중·하, part1…)은 한 문서로 합치고"
      "(`merge_parts`), 대본이 아닌 파일·같은 글 안 중복 파일·압축 파일 자체는 제외 행으로 남긴다.")
    w("")
    posts = {r["post_id"] for r in rows}
    w(f"- 게시글 {len(posts)}개, catalog.csv {len(rows)}행 = 문서 {len(docs)}개 + 제외 {len(excl)}개")
    w(f"- 파일이 둘 이상인 문서 {sum(len(r['files_list']) > 1 for r in docs)}개, 압축에서 나온 문서 "
      f"{sum(any(f.startswith('extracted/') for f in r['files_list']) for r in docs)}개, "
      f"한 게시글에서 문서가 둘 이상 나온 글 {sum(v > 1 for v in Counter(r['post_id'] for r in docs).values())}개")
    exn = Counter((r["note"] or "").split(":")[0] for r in excl)
    why_ko = {"duplicate_file_of": "같은 글 안에서 내용이 같은 파일(sha1 일치)", "archive_container": "압축 파일 자체(풀린 파일이 문서가 됨)",
              "not_script": "대본으로 읽을 수 없는 형식(실행 파일·그림·자막 파일 등)"}
    w("")
    L.extend(table(["제외 사유(`note`)", "뜻", "행"], [[f"`{k}`", why_ko.get(k, ""), v] for k, v in exn.most_common()]))
    w("")

    # ---------- 3
    w("## 3. 글자를 얻는 경로")
    w("")
    w("형식에 따라 글자를 얻는 길이 갈린다. 문서별 값은 `types.csv` 의 `route` 칸.")
    w("")
    by_route: dict[str, list[dict]] = defaultdict(list)
    for t in T:
        by_route[t["route"]].append(t)
    body = []
    known = [k for k, *_ in ROUTES]
    for k, ko, how, code in ROUTES:
        ts = by_route.get(k, [])
        if not ts:
            continue
        body.append([f"`{k}`", ko, how, len(ts), counts(Counter(t["real_format"] for t in ts)), code])
    for k, ts in sorted(by_route.items()):
        if k not in known:
            body.append([f"`{k}`", "여러 경로(파일마다 다름)", "", len(ts), counts(Counter(t["real_format"] for t in ts)), ""])
    L.extend(table(["경로", "이름", "방법", "문서", "원본 형식", "코드"], body))
    w("")
    meth = Counter((x["real_format"], x["method"]) for x in log.values() if x.get("ok"))
    w("- 변환 방법(파일 기준, cache/convert_log.jsonl): " + ", ".join(f"{f} {m} {n}" for (f, m), n in sorted(meth.items())))
    whys = Counter(wy for t in T for wy in t["ocr_why"].split("+") if wy)
    why_ko = {"pdf_scan": "스캔 PDF", "textlayer_broken": "원본 PDF 글자층이 깨짐", "converted_textlayer_broken": "한컴 변환 PDF 글자층이 깨짐",
              "pdf_broken": "글자층을 읽을 수 없는 PDF"}
    w("- OCR 사유(문서 기준): " + ", ".join(f"{why_ko.get(k, k)} {v}" for k, v in whys.most_common()))
    two = [t for t in T if t["two_up"] == "true"]
    w(f"- 두 쪽 펼침(한 장에 대본 두 쪽을 나란히 인쇄) 문서 {len(two)}개: 가운데 빈 띠에서 왼쪽·오른쪽 쪽으로 나눈다"
      f"(`lines._gutter`, OCR 은 `ocr.split_two_up`). 경로별 " + counts(Counter(t["route"] for t in two)))
    resp = [t for t in T if "source" in t["spacing"].split("+")]
    skip = [t for t in T if "skipped" in t["spacing"].split("+")]
    w(f"- 공백 복원(`types.csv` 의 `spacing`, lines.jsonl 머리 행 `made.spacing`): 한컴 PDF 글자층에 띄어쓰기·이름 뒤 탭이 "
      f"빠진 파일(공백 손실 0.15 초과)은 줄 글의 공백·탭을 원본 문서 글에 맞춘다(`lines.restore_spacing`). "
      f"`source` {len(resp)}문서"
      + (" (" + ", ".join(f"{t['doc_id']} {'/'.join(f'{x:.3f}' for x in t['_spacing_loss'] if x is not None)}"
                          for t in resp) + ")" if resp else "")
      + f", `skipped`(원본 글을 못 읽어 견주지 못함) {len(skip)}문서"
      + (" (" + ", ".join(t["doc_id"] for t in skip) + ")" if skip else "")
      + ". 경로별 " + counts(Counter(t["route"] for t in resp)))
    w("- 줄을 만든 lines 판(머리 행 `made.version`, 공백 손실은 0.4.0 부터 잰다): "
      + counts(Counter(t["_lines_version"] or "-" for t in T)))
    w("")

    # ---------- 4
    w("## 4. 내용 종류·언어·판본")
    w("")
    content = Counter(c.split(":")[0] if c.startswith("tv_") else c for t in T for c in t["content"].split("|") if c)
    content_ko = {"content:subtitle_transcript": "자막 받아쓰기(시각 목록)", "content:synopsis": "시놉시스", "content:transcript": "영화 받아쓰기",
                  "content:recording_script_xls": "녹음 대본(엑셀)", "tv_episode": "드라마 한 회", "tv_series": "드라마 여러 회"}
    w(f"- 내용 종류(`note`, 원본을 직접 보고 정한 `catalog.POST_OVERRIDES`·`FILE_OVERRIDES`): 영화 대본이 아닌 문서 "
      f"{sum(bool(t['content']) for t in T)}개: " + ", ".join(f"{content_ko.get(k, k)} {v}" for k, v in content.most_common()))
    w("- 대본 언어(`script_lang`, `review.detect_lang` 이 글자 비율과 불용어로 판정): " + counts(Counter(t["script_lang"] for t in T)))
    w(f"- 번역 대본(`is_translation`, 외국영화의 한국어 대본 또는 한국영화의 외국어 대본): {sum(t['is_translation'] == 'true' for t in T)}개")
    w("- 판본(`version_label`, 제목·첨부 이름의 단서 `catalog.VERSION_CUES`): " + counts(Counter(t["version_label"] for t in T)))
    films = {r["film_id"] for r in docs if r["film_id"]}
    w(f"- 영화 {len(films)}편, 영화마다 대표 대본(`is_primary`) 하나: 대표 {sum(t['is_primary'] == 'true' for t in T)}개, "
      f"같은 영화의 다른 판 {sum(t['is_primary'] != 'true' for t in T)}개")
    lang_cat = Counter((cat_of[t["doc_id"]], t["script_lang"]) for t in T)
    w("- 분류 폴더별 언어: " + ", ".join(f"{CAT_KO.get(c, c)} {lg} {n}" for (c, lg), n in sorted(lang_cat.items(), key=lambda kv: (kv[0][0] != 'ko_film', -kv[1]))))
    w("")

    # ---------- 5
    w("## 5. 대본 꼴")
    w("")
    w("### 5.1 대사 표기(이름을 적는 법)")
    w("")
    w(f"문서의 대표 표기는 script.json 대사 요소 가운데 이름이 적힌 것(앞 대사에서 이어받은 화자 제외)의 첫머리를 보고 가장 많은 표기로 정했다"
      f"(이름 적힌 대사가 {MIN_DIALOGUES}개 미만이면 `none`). 문서별 값은 `types.csv` 의 `dialogue_form`, 그 표기의 비율은 `form_share`.")
    w("")
    body = []
    for k, ko, how, ex, code in FORMS:
        ts = [t for t in T if t["dialogue_form"] == k]
        if not ts:
            continue
        n, kf, ff = split(t["doc_id"] for t in ts)
        shares = sorted(float(t["form_share"]) for t in ts if t["form_share"])
        med = f"{shares[len(shares) // 2]:.2f}" if shares else "-"
        nr = sum(bool(t["needs_review"]) for t in ts)
        body.append([f"`{k}`", ko, how, ex, n, kf, ff, med, nr, code])
    L.extend(table(["값", "표기", "모양", "보기(지어낸 글)", "문서", "한국영화", "외국영화", "표기 비율 중앙값", "확인 필요 문서", "파서 분기"], body))
    w("")
    own = [t for t in T if t["dialogue_form"] == "own_line"]
    w(f"- `own_line` {len(own)}개 가운데 들여쓰기 없는 이름 줄(`flat_names`) {sum('flat_names' in t['parser_branches'].split('|') for t in own)}개, "
      f"대사가 지문 여백에 놓인 대문자 이름 줄(`collapsed_cues`) {sum('collapsed_cues' in t['parser_branches'].split('|') for t in own)}개, "
      f"나머지는 들여쓰거나 가운데 놓은 이름 줄")
    deco = Counter(t["name_deco"] for t in T if t["name_deco"])
    w(f"- 이름 꾸밈(`name_deco`, 이름 적힌 대사의 {int(DECO_MIN_SHARE * 100)}% 이상): "
      + (", ".join(f"{DECO_KO.get(k, k)} {v}" for k, v in deco.most_common()) or "없음"))
    mixed = [t for t in T if t["form_share"] and float(t["form_share"]) < 0.8]
    w(f"- 한 문서에 표기가 섞인 문서(대표 표기 비율 0.8 미만) {len(mixed)}개: " + counts(Counter(t["dialogue_form"] for t in mixed)))
    w("")
    w("### 5.2 파서가 켠 특수 분기")
    w("")
    w("규칙 파서가 문서마다 `quality.notes` 에 적은 분기다(한 문서에 여럿 가능). 문서별 값은 `types.csv` 의 `parser_branches`.")
    w("")
    body = []
    for k, _, ko, code in BRANCHES:
        ts = [t for t in T if k in t["parser_branches"].split("|")]
        if ts:
            n, kf, ff = split(t["doc_id"] for t in ts)
            body.append([f"`{k}`", ko, n, kf, ff, code])
    L.extend(table(["값", "뜻", "문서", "한국영화", "외국영화", "코드"], body))
    w("")
    w("### 5.3 씬 머리")
    w("")
    sc = [t["scenes"] for t in T if t["scenes"] != ""]
    zero = [t for t in T if t["scenes"] == 0]
    scs = sorted(sc)
    w(f"- 씬 머리를 하나도 못 찾은 문서 {len(zero)}개(대표 표기별 " + counts(Counter(t["dialogue_form"] for t in zero))
      + f"), 문서당 씬 수 중앙값 {scs[len(scs) // 2] if scs else '-'}")
    w("")
    w("### 5.4 유형별 확인 필요(qa/needs_review.csv)")
    w("")
    probs = sorted({p for t in T for p in t["needs_review"].split("|") if p})
    if probs:
        body = []
        for k, ko, *_ in FORMS:
            ts = [t for t in T if t["dialogue_form"] == k]
            if not ts:
                continue
            body.append([f"`{k}`", len(ts)] + [sum(p in t["needs_review"].split("|") for t in ts) for p in probs])
        L.extend(table(["대표 표기", "문서"] + [f"`{p}`" for p in probs], body))
        w("")
        body = []
        for k, ts in sorted(by_route.items(), key=lambda kv: -len(kv[1])):
            body.append([f"`{k}`", len(ts)] + [sum(p in t["needs_review"].split("|") for t in ts) for p in probs])
        L.extend(table(["글자 경로", "문서"] + [f"`{p}`" for p in probs], body))
        w("")
    n_llm = sum(t["_llm_now"] for t in T)
    w("- 유형을 센 판(`parser`): " + counts(Counter(t["parser"] for t in T))
      + (f". docs/ 에서 지금 LLM 판인 문서 {n_llm}개는 보관된 규칙 판으로 셌다" if rules_only and n_llm else ""))
    w("")

    # ---------- 6
    w("## 6. types.csv 칸")
    w("")
    col_ko = {
        "doc_id": "문서 ID(catalog.csv 와 같음)", "category": "분류 폴더(ko_film, foreign_film)", "title": "정리한 제목(title_norm)",
        "real_format": "파일 형식(1절)", "files": "파일 수", "route": "글자를 얻은 경로(3절, 파일마다 다르면 + 로 이음)",
        "convert_method": "변환 방법(convert_log 의 method)", "ocr_why": "OCR 한 까닭", "two_up": "두 쪽 펼침",
        "script_lang": "대본 언어", "is_translation": "번역 대본", "version_label": "판본", "content": "영화 대본이 아닌 내용 종류",
        "is_primary": "영화의 대표 대본", "parser": "유형을 센 판을 만든 파서(`--rules` 면 늘 규칙 파서)", "lines": "줄 수", "scenes": "씬 수",
        "dialogues": "대사 요소 수",
        "dialogue_form": "대표 대사 표기(5.1절)", "form_share": "이름 적힌 대사 중 대표 표기의 비율", "name_deco": "이름 꾸밈(letterspaced, bullet)",
        "parser_branches": "파서가 켠 특수 분기(5.2절, 세로선으로 이음)", "score": "유형을 센 판의 품질 점수(quality.score)",
        "needs_review": "확인 필요 문제(qa/needs_review.csv 그대로, 세로선으로 이음)",
        "spacing": "공백 복원(3절): " + ", ".join(f"`{k}` {v}" for k, v in SPACING_KO.items())
                   + ", 빈 값은 기록 없음(견줬고 문턱 아래, 한컴 PDF 경로가 아님, 또는 공백 손실을 재기 전의 lines 판)",
    }
    L.extend(table(["칸", "뜻"], [[f"`{c}`", col_ko[c]] for c in CSV_FIELDS]))

    buf = io.StringIO()
    cw = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    cw.writeheader()
    for t in T:
        cw.writerow({k: util.nfc(str(t[k])) for k in CSV_FIELDS})
    util.write_text(out_dir / "types.csv", buf.getvalue(), encoding="utf-8-sig")
    util.write_text(out_dir / "TYPES.md", "\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
