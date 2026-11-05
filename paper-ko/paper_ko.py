"""paper_ko - 영어 논문 PDF를 글자 자리 그대로 한국어로 바꿔 넣는 번역기.

사용:
  python paper_ko.py 논문.pdf                  # Claude(구독 로그인)로 번역, 품질 좋음
  python paper_ko.py 논문.pdf --engine google  # 무료 구글 번역, 빠르지만 품질 낮음
  python paper_ko.py 논문.pdf --pages 1-3,5    # 일부 쪽만
결과(원본 옆에 생성):
  논문_ko.pdf       번역본 (그림·표·수식·배치는 원본 그대로, 글자만 한국어)
  논문_ko_dual.pdf  왼쪽 원문 | 오른쪽 번역 나란히
"""
import argparse
import concurrent.futures as cf
import hashlib
import html as H
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FROZEN = getattr(sys, "frozen", False)  # exe 로 묶였을 때
# 번역 저장소·설정 위치 (exe 는 임시 폴더에서 돌아서 %LOCALAPPDATA% 에 둔다)
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "paper-ko" if FROZEN else HERE
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # claude 부를 때 까만 창 안 뜨게


class Cancelled(Exception):
    """사용자가 취소함."""


def find_claude():
    """설치된 Claude Code(claude.exe) 경로. 없으면 None."""
    cands = [shutil.which("claude.exe"), shutil.which("claude"),
             Path.home() / ".local" / "bin" / "claude.exe",
             Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "claude.exe"]
    npm = Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "@anthropic-ai"
    if npm.exists():
        cands += sorted(npm.glob("**/claude.exe"))
    for c in cands:
        if c and str(c).lower().endswith(".exe") and Path(c).exists():
            return str(c)
    return None


ENV_KEY = "CLAUDE_CODE_OAUTH_TOKEN"
ENV_TEMPLATE = """# 논문 번역기 — Claude 구독 토큰
# 'claude setup-token' 으로 받은 토큰을 = 뒤에 붙여넣고 저장하세요.
# 이 프로그램은 아래 한 줄만 읽습니다. 이 파일은 남에게 주지 마세요.
CLAUDE_CODE_OAUTH_TOKEN=
"""


def env_file_path():
    """exe(또는 스크립트) 옆의 .env"""
    return (Path(sys.executable).parent if FROZEN else HERE) / ".env"


def env_file_token(path=None):
    """.env 에서 CLAUDE_CODE_OAUTH_TOKEN 한 줄만 읽는다.
    같은 파일에 ANTHROPIC_API_KEY 등이 있어도 읽지 않는다 (API 요금 방지)."""
    try:
        text = Path(path or env_file_path()).read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        if sep and key.strip() == ENV_KEY:
            val = val.split(" #")[0].strip().strip('"').strip("'").strip()
            return val or None
    return None


def claude_env(token=None):
    """API 키 변수는 지운다(요금 방지). 토큰을 주면 Claude Code 공식 변수로만 넘긴다(디스크에 저장 안 함)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN" if token else "")}
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def claude_status(exe, token=None):
    """claude auth status (사용량 안 씀). dict 또는 None."""
    try:
        r = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, text=True, encoding="utf-8",
                           env=claude_env(token), stdin=subprocess.DEVNULL, timeout=30,
                           cwd=tempfile.gettempdir(), creationflags=NO_WINDOW)
        return json.loads(r.stdout)
    except Exception:
        return None


def check_claude(exe, token=None):
    """실제로 짧은 요청 한 번 (haiku). (성공 여부, 메시지)"""
    cmd = [exe, "-p", "--model", "haiku", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--output-format", "json"]
    try:
        r = subprocess.run(cmd, input="Reply with exactly: OK", capture_output=True, text=True, encoding="utf-8",
                           env=claude_env(token), timeout=120, cwd=tempfile.gettempdir(), creationflags=NO_WINDOW)
    except Exception as e:
        return False, str(e)[:300]
    try:
        out = json.loads(r.stdout)
    except ValueError:
        return False, (r.stderr or r.stdout or "응답 없음").strip()[:300]
    if out.get("is_error"):
        return False, str(out.get("result"))[:300]
    return True, "연결됨"


def pick_font(*names):
    for n in names:
        if (FONT_DIR / n).exists():
            return n
    sys.exit(f"한글 글꼴을 찾지 못했습니다: {names} ({FONT_DIR})")


SERIF = pick_font("NanumMyeongjo.ttf", "KoPubBatangMedium.ttf", "HANBatang.ttf", "malgun.ttf")
SERIF_B = pick_font("NanumMyeongjoBold.ttf", "KoPubBatangBold.ttf", "HANBatangB.ttf", "malgunbd.ttf")
SANS = pick_font("NanumGothic.ttf", "NotoSansKR-Regular.ttf", "malgun.ttf")
SANS_B = pick_font("NanumGothicBold.ttf", "NotoSansKR-Bold.ttf", "malgunbd.ttf")
MATH = pick_font("times.ttf", "malgun.ttf")
MATH_I = pick_font("timesi.ttf", "times.ttf", "malgun.ttf")

CSS = f"""
@font-face {{font-family: kos; src: url({SERIF});}}
@font-face {{font-family: kos; src: url({SERIF_B}); font-weight: bold;}}
@font-face {{font-family: kog; src: url({SANS});}}
@font-face {{font-family: kog; src: url({SANS_B}); font-weight: bold;}}
@font-face {{font-family: mth; src: url({MATH});}}
@font-face {{font-family: mth; src: url({MATH_I}); font-style: italic;}}
body, p, div, span {{margin: 0; padding: 0;}}
.m {{font-family: mth; font-weight: normal;}}
"""

# 수식 글꼴 (TeX Computer Modern 수식체, AMS 기호, 각종 Math 글꼴)
MATH_FONT = re.compile(
    r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|EUEX|EURM|RSFS|WASY|STMARY|BBM|DSROM|"
    r"MATH|SYMBOL|MTSY|MTMI|MTEX|TXSY|TXEX|TXMI|PXSY|PXEX|PXMI|RTXMI|ESINT", re.I)
# Computer Modern 본문체 (본문이 Times 계열이면 이 글꼴은 수식 안의 숫자·기호)
CM_ROMAN = re.compile(r"^(CMR|CMBX|CMSL|SFRM|SFBX|LMRoman|CMU)", re.I)
BOLD_FONT = re.compile(r"Bold|Medi|Black|Semibold|Demi|Heavy|CMBX|SFBX|[.-]B$", re.I)
SANS_FONT = re.compile(r"Sans|Helvet|Arial|Gothic|Calibri|Verdana|Segoe|Roboto|Lato|NimbusSan|"
                       r"CMSS|SFSS|Futura|Avenir|Gill|Myriad", re.I)
COMPOUND_HEAD = re.compile(
    r"\b(self|non|multi|state|of|the|end|cross|semi|pre|post|well|fine|high|low|long|short|large|small|"
    r"real|single|two|three|top|zero|one|few|co|sub|inter|intra|data|task|model|domain|sequence|"
    r"token|word|level|layer|time|point|channel|pixel|class|label|object|image|text|over|under|"
    r"out|in|on|off|up|down|by|half|full|first|second|third|higher|lower)-\s*$", re.I)
REF_HEADS = {"references", "reference", "bibliography", "literaturecited", "citedliterature"}

FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
PH = re.compile(r"[{｛]\s*(\d+)\s*[}｝]")  # 수식 자리표시 {1}
# 글자 대응표(ToUnicode)가 없는 TeX 큰 기호 글꼴은 알파벳으로 뽑힌다 → 원래 기호로
CMEX_MAP = {"P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫", "H": "∮", "I": "∮",
            "S": "∐", "p": "√", "q": "√", "r": "√", "s": "√", "t": "√", "v": "√"}


def clean_font(name):
    return re.sub(r"^[A-Z]{6}\+", "", name or "")


# ───────────────────────────── 추출 ─────────────────────────────

@dataclass
class Para:
    pno: int
    rect: pymupdf.Rect
    erase: list
    nrows: int
    pitch: float
    size: float
    color: int
    bold: bool
    sans: bool
    src: str
    maths: list
    lead: str
    math_chars: int
    text_chars: int
    row_x: list = field(default_factory=list)  # 각 줄 (x0, x1)
    tight: pymupdf.Rect = None                  # 실제 글자 영역 (rect 는 아래로 여유를 둔 것)
    align: str = "justify"
    translate: bool = True
    why: str = ""
    ko: str = ""
    lead_ko: str = ""

    @property
    def plain(self):
        return re.sub(r"\s+", " ", PH.sub(" ", self.src)).strip()

    @property
    def words(self):
        return re.findall(r"[A-Za-z]{2,}", self.plain)


class FontInfo:
    def __init__(self, dicts):
        chars = Counter()
        sizes = Counter()
        for d in dicts.values():
            for b in d["blocks"]:
                for l in b.get("lines", []):
                    for s in l["spans"]:
                        n = len(s["text"].strip())
                        chars[clean_font(s["font"])] += n
                        sizes[round(s["size"] * 2) / 2] += n
        self.body_font = chars.most_common(1)[0][0] if chars else ""
        self.body_size = sizes.most_common(1)[0][0] if sizes else 10.0
        self.cm_body = bool(CM_ROMAN.match(self.body_font))

    def is_math(self, font):
        if MATH_FONT.search(font):
            return True
        return (not self.cm_body) and bool(CM_ROMAN.match(font))


def line_base(l):
    spans = [s for s in l["spans"] if s["text"].strip()]
    big = max(s["size"] for s in spans)
    main = max((s for s in spans if s["size"] >= 0.85 * big), key=lambda s: len(s["text"]))
    return main["origin"][1], big


def group_rows(lines):
    """같은 기준선(baseline)의 line 들을 한 줄(row)로 묶는다. 위·아래첨자 line 도 본 줄에 붙는다.
    (bbox 가운데로 묶으면 루트 기호처럼 키 큰 글자 때문에 두 줄이 하나로 합쳐진다)"""
    rows = []
    for l in sorted(lines, key=lambda l: line_base(l)[0]):
        base, size = line_base(l)
        if rows:
            r = rows[-1]
            same = abs(base - r["base0"]) < 0.35 * max(r["size0"], size)
            touches = l["bbox"][3] > r["base0"] - r["size0"] and l["bbox"][1] < r["base0"]
            script = size < 0.85 * r["size0"] and touches
            # 문장 속 분수의 분모(예: √d_k)처럼 짧고 본 줄에 걸친 조각
            narrow = touches and (l["bbox"][2] - l["bbox"][0]) < 4 * r["size0"]
            # 반대로: 작은 조각(위첨자·분자)이 먼저 줄을 만들었고 지금 본 줄이 그 위에 걸침
            ry0 = min(q["bbox"][1] for q in r["lines"])
            ry1 = max(q["bbox"][3] for q in r["lines"])
            under = r["size0"] < 0.85 * size and ry1 > base - size and ry0 < base
            if same or script or narrow or under:
                r["lines"].append(l)
                if size > r["size0"]:
                    r["base0"], r["size0"] = base, size
                continue
        rows.append({"lines": [l], "base0": base, "size0": size})
    for r in rows:
        r["lines"].sort(key=lambda l: l["bbox"][0])
        r["x0"] = min(l["bbox"][0] for l in r["lines"])
        r["x1"] = max(l["bbox"][2] for l in r["lines"])
        r["y0"] = min(l["bbox"][1] for l in r["lines"])
        r["y1"] = max(l["bbox"][3] for l in r["lines"])
        spans = [s for l in r["lines"] for s in l["spans"] if s["text"].strip()]
        big = max(s["size"] for s in spans)
        main = max((s for s in spans if s["size"] >= 0.85 * big), key=lambda s: len(s["text"]))
        r["size"], r["base"] = main["size"], main["origin"][1]
        r["text"] = "".join(s["text"] for s in spans)
        r["allbold"] = all(BOLD_FONT.search(clean_font(s["font"])) or s["flags"] & 16 for s in spans)
    return rows


def split_paragraphs(rows):
    """한 블록 안에 문단이 여러 개 섞여 있으면 나눈다 (들여쓰기, 짧은 끝줄, 글자 크기·간격 변화)."""
    xl = min(r["x0"] for r in rows)
    xr = max(r["x1"] for r in rows)
    out = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        s = prev["size"]
        ends = bool(re.search(r"[.:?!][\"')\]]?\s*$", prev["text"]))
        indent = cur["x0"] - xl > 0.8 * s
        short_prev = prev["x1"] < xr - 2 * s
        size_change = abs(cur["size"] - prev["size"]) > 0.8
        gap = cur["y0"] - prev["y1"] > 0.7 * s
        head = prev["allbold"] and not cur["allbold"] and len(prev["text"]) < 80
        # 시작 위치가 크게 다르고 가운데 정렬도 아니면 다른 덩어리 (예: "(b) 라벨" 다음 줄 "Figure 3:")
        shifted = (abs(prev["x0"] - cur["x0"]) > 4 * s
                   and abs((prev["x0"] + prev["x1"]) - (cur["x0"] + cur["x1"])) / 2 > 2 * s)
        if size_change or gap or head or shifted or (ends and (indent or short_prev)):
            out.append([cur])
        else:
            out[-1].append(cur)
    return out


def build_para(rows, pno, fi: FontInfo):
    pieces = []  # [kind, text, html, bold]  kind: t(본문) m(수식) j(줄바꿈 이음)
    erase = []
    colors, fonts, sizes = Counter(), Counter(), Counter()
    for ri, row in enumerate(rows):
        prev_x1 = None
        for l in row["lines"]:
            if prev_x1 is not None:
                if l["bbox"][0] - prev_x1 > 0.15 * row["size"]:
                    pieces.append(["t", " ", None, False])
            prev_x1 = l["bbox"][2]
            for s in l["spans"]:
                t = s["text"]
                if not t:
                    continue
                font = clean_font(s["font"])
                if t.strip():
                    erase.append(pymupdf.Rect(s["bbox"]))
                script = None
                if s["size"] < 0.85 * row["size"]:
                    dy = s["origin"][1] - row["base"]
                    if dy < -0.15 * row["size"]:
                        script = "sup"
                    elif dy > 0.08 * row["size"]:
                        script = "sub"
                if (fi.is_math(font) or script) and t.strip():
                    tt = unicodedata.normalize("NFKC", t).replace("\ufffd", "")
                    if "CMEX" in font.upper():
                        tt = "".join(CMEX_MAP.get(c, c) for c in tt)
                    h = H.escape(tt)
                    if re.search(r"CMMI|MathItalic|TXMI|PXMI|RTXMI|MTMI", font, re.I) or (s["flags"] & 2 and MATH_FONT.search(font)):
                        h = f"<i>{h}</i>"
                    if script:
                        h = f"<{script} style='font-size:70%'>{h}</{script}>"
                    pieces.append(["m", tt, h, False])
                else:
                    bold = bool(BOLD_FONT.search(font) or s["flags"] & 16)
                    pieces.append(["t", t, None, bold])
                    n = len(t.strip())
                    colors[s["color"]] += n
                    fonts[font] += n
                    sizes[round(s["size"], 1)] += n
        if ri < len(rows) - 1:
            pieces.append(["j", "", None, False])

    # 줄바꿈 이음: 하이픈으로 끊긴 단어는 붙이고, 나머지는 공백
    for i, p in enumerate(pieces):
        if p[0] != "j":
            continue
        before = next((q for q in reversed(pieces[:i]) if q[0] != "j" and q[1].strip()), None)
        after = next((q for q in pieces[i + 1:] if q[0] != "j" and q[1].strip()), None)
        p[0] = "t"
        if (before and after and before[0] == "t" and after[0] == "t"
                and re.search(r"[A-Za-z]-\s*$", before[1]) and re.match(r"\s*[a-z]", after[1])):
            # "connec-/tion" 은 붙이고, "self-/attention" 같은 합성어는 하이픈을 살려 붙인다
            if not COMPOUND_HEAD.search(before[1]):
                before[1] = re.sub(r"-\s*$", "", before[1])
            p[1] = ""
        else:
            p[1] = " "

    # 앞머리 굵은 글씨 (예: "Encoder:")는 따로 번역해 굵게 표시
    text_chars = sum(len(p[1].strip()) for p in pieces if p[0] == "t")
    bold_chars = sum(len(p[1].strip()) for p in pieces if p[0] == "t" and p[3])
    para_bold = text_chars > 0 and bold_chars / text_chars > 0.6
    lead = ""
    if not para_bold:
        k = 0
        while k < len(pieces) and pieces[k][0] == "t" and (pieces[k][3] or not pieces[k][1].strip()):
            k += 1
        cand = "".join(p[1] for p in pieces[:k]).strip()
        if cand and 0 < len(cand) <= 60 and k < len(pieces) and any(p[1].strip() for p in pieces[k:]):
            lead, pieces = cand, pieces[k:]

    # 수식 조각 이어 붙이기 → {n} 자리표시
    src, maths, math_chars = [], [], 0
    i = 0
    while i < len(pieces):
        p = pieces[i]
        if p[0] != "m":
            src.append(p[1])
            i += 1
            continue
        htmls, texts = [], []
        while i < len(pieces):
            q = pieces[i]
            if q[0] == "m":
                htmls.append(q[2]); texts.append(q[1]); i += 1
            elif not q[1].strip() and i + 1 < len(pieces) and pieces[i + 1][0] == "m" and q[1]:
                htmls.append(" "); texts.append(" "); i += 1
            else:
                break
        maths.append("".join(htmls))
        math_chars += len("".join(texts).replace(" ", ""))
        src.append(f"{{{len(maths)}}}")
    src = re.sub(r"\s+", " ", "".join(src)).strip()

    x0 = min(r["x0"] for r in rows)
    x1 = max(r["x1"] for r in rows)
    y0 = rows[0]["y0"]
    size = sizes.most_common(1)[0][0] if sizes else rows[0]["size"]
    if len(rows) > 1:
        pitch = (rows[-1]["base"] - rows[0]["base"]) / (len(rows) - 1)
    else:
        pitch = rows[0]["y1"] - rows[0]["y0"]
    pitch = max(pitch, size * 1.05)
    y1 = max(rows[-1]["y1"], y0 + pitch * len(rows)) + 0.2 * size
    main_font = fonts.most_common(1)[0][0] if fonts else ""
    return Para(
        pno=pno, rect=pymupdf.Rect(x0, y0, x1, y1), erase=erase, nrows=len(rows), pitch=pitch,
        size=size, color=colors.most_common(1)[0][0] if colors else 0, bold=para_bold,
        sans=bool(SANS_FONT.search(main_font)), src=src, maths=maths, lead=lead,
        math_chars=math_chars, text_chars=text_chars,
        row_x=[(r["x0"], r["x1"]) for r in rows], tight=pymupdf.Rect(x0, y0, x1, rows[-1]["y1"]))


def extract_paras(d, pno, fi):
    paras = []
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        lines = [l for l in b["lines"] if any(s["text"].strip() for s in l["spans"])]
        # 세로·회전 글자 (예: arXiv 옆 도장)는 건드리지 않는다
        lines = [l for l in lines if abs(l["dir"][1]) < 0.01 and l["dir"][0] > 0]
        if not lines:
            continue
        rows = group_rows(lines)
        cells = split_cells(rows)
        if cells:  # 표·그림 라벨처럼 가로로 크게 띄어 쓴 블록 → 칸마다 한 줄 문단
            for cell in cells:
                paras.append(build_para(group_rows(cell), pno, fi))
        else:
            for prows in split_paragraphs(rows):
                paras.append(build_para(prows, pno, fi))
    return paras


def split_cells(rows):
    """어느 줄이든 글자 사이가 글자 2.5개 넘게 비어 있으면 칸으로 나눈다. 없으면 None."""
    cells, found = [], False
    for r in rows:
        cur = [r["lines"][0]]
        for l in r["lines"][1:]:
            if l["bbox"][0] - cur[-1]["bbox"][2] > 2.5 * r["size"]:
                cells.append(cur)
                cur, found = [], True
            cur.append(l)
        cells.append(cur)
    return cells if found else None


# ───────────────────────────── 번역 대상 고르기 ─────────────────────────────

class State:
    in_refs = False


def figure_regions(page):
    W, Hh = page.rect.width, page.rect.height
    regs = []
    for im in page.get_image_info():
        r = pymupdf.Rect(im["bbox"]) & page.rect
        if r.width > 20 and r.height > 20 and r.get_area() < 0.8 * W * Hh:
            regs.append(r)
    try:
        for r in page.cluster_drawings():
            if r.width > 30 and r.height > 30 and r.get_area() < 0.8 * W * Hh:
                regs.append(r)
    except Exception:
        pass
    return regs


def overlap(a, b):
    i = a & b
    return 0 if i.is_empty else i.get_area() / max(a.get_area(), 1e-6)


def decide(page, paras, fi, state):
    regs = figure_regions(page)
    body_paras = [p for p in paras if p.nrows >= 3]
    margins = [p.rect.x0 for p in body_paras]

    for p in paras:
        plain, words = p.plain, p.words
        nonspace = len(re.sub(r"\s", "", plain))
        alpha = sum(c.isalpha() for c in plain)
        key = re.sub(r"[\d.IVX\s]+", "", plain[:40]).lower().rstrip(":")
        if not words:
            p.translate, p.why = False, "글자없음"
        elif p.size < 2:  # 안 보이는 글자층 (OCR 층·숨은 워터마크)
            p.translate, p.why = False, "작은글자"
        elif re.search(r"[가-힣]", plain):
            p.translate, p.why = False, "이미한글"
        elif p.math_chars > 0.5 * (p.math_chars + nonspace):
            p.translate, p.why = False, "수식"
        elif alpha < 0.45 * max(nonspace, 1):
            p.translate, p.why = False, "숫자표"
        elif re.fullmatch(r"[\w.@:/\-~%?=&#,;{} ]+", plain) and re.search(r"@|https?:|www\.", plain):
            p.translate, p.why = False, "주소"
        elif key in REF_HEADS:
            state.in_refs = True
        elif state.in_refs:
            appx = p.nrows == 1 and (p.bold or p.size > fi.body_size * 1.1) and (
                re.match(r"(appendix|supplementary)", plain, re.I) or re.match(r"[A-H](\.\d+)*\.?\s+[A-Z][a-z]", plain))
            if appx:
                state.in_refs = False
            else:
                p.translate, p.why = False, "참고문헌"
        if not p.translate:
            continue
        if len(words) < 15 and any(overlap(p.rect, r) > 0.5 for r in regs):
            p.translate, p.why = False, "그림안"

    # 표 칸·그림 글자: 같은 높이에 짧은 한 줄짜리가 셋 이상 나란히 있으면 표로 본다
    # (옆 단 그림 속 글자는 세지 않는다 — 2단 논문의 소제목이 표로 오인되지 않게)
    singles = [p for p in paras if p.nrows == 1 and not any(overlap(p.rect, r) > 0.5 for r in regs)]
    for p in singles:
        if not p.translate or len(p.words) > 6:
            continue
        same_row = 0
        for q in singles:
            if q is p:
                continue
            ov = min(p.rect.y1, q.rect.y1) - max(p.rect.y0, q.rect.y0)
            if ov > 0.5 * min(p.rect.height, q.rect.height) and (q.rect.x0 > p.rect.x1 or q.rect.x1 < p.rect.x0):
                same_row += 1
        if same_row >= 2:
            p.translate, p.why = False, "표"

    # 정렬
    for p in paras:
        if p.nrows == 1:
            at_margin = any(-2 < p.rect.x0 - m < 20 for m in margins)
            p.align = "left" if at_margin or not margins else "center"
        else:
            centers = [(a + b) / 2 for a, b in p.row_x]
            widths = [b - a for a, b in p.row_x]
            centered = max(centers) - min(centers) < 3 and min(widths) < 0.9 * max(widths)
            p.align = "center" if centered else "justify"


def skip_front_matter(paras):
    """첫 쪽 제목 아래 ~ Abstract 위 (저자·소속·이메일)는 그대로 둔다."""
    # "Abstract—" 처럼 굵은 머리(lead)로 떨어진 경우도 잡는다
    abs_p = next((p for p in paras if re.match(r"\W*abstract\b", f"{p.lead} {p.plain}".strip(), re.I)), None)
    if not abs_p:
        return
    above = [p for p in paras if p.rect.y1 <= abs_p.rect.y0 + 1]
    if not above:
        return
    title = max(above, key=lambda p: p.size)
    for p in above:
        if p.rect.y0 >= title.rect.y1 - 1 and p.size < 0.9 * title.size and p.translate:
            p.translate, p.why = False, "저자"


# ───────────────────────────── 번역기 ─────────────────────────────

SYSTEM_PROMPT = """너는 영어 학술 논문을 한국어로 옮기는 전문 번역가다.
입력은 JSON 객체 {"paper": 논문 맥락, "items": [문자열, ...]} 이다. items 의 각 문자열을 한국어로 번역한다.

규칙
1. 자연스러운 한국어 논문 문체(~다, ~한다)로 쓴다. 원문보다 길어지지 않게 군더더기 없이 쓴다.
2. {1}, {2} 같은 중괄호 숫자 토큰은 수식 자리표시다. 하나도 빠뜨리거나 더하지 말고 그대로 두되, 한국어 어순에 맞는 자리에 둔다.
3. 인용 번호([12], (Smith et al., 2020)), 숫자, 단위, 사람 이름, 모델·데이터셋·지표·알고리즘 이름(Transformer, ImageNet, BLEU 등), 약어, 코드, URL 은 원문 그대로 둔다.
4. 전문 용어는 한국 학계에서 통용되는 번역어를 쓰고(예: residual connection → 잔차 연결), 통용 번역어가 없으면 영어 그대로 둔다. 괄호 영어 병기는 하지 않는다.
5. 제목·소제목·그림 캡션 같은 짧은 항목은 짧게 옮긴다. "Figure 3:" → "그림 3:", "Table 2:" → "표 2:".
6. 문장이 중간에서 잘려 있어도(쪽·단 넘김) 잘린 그대로 자연스럽게 옮긴다.
7. 출력은 번역문 문자열만 담은 JSON 배열 하나다. 개수와 순서는 items 와 같다. 설명, 코드블록 표시(```)는 쓰지 않는다."""


class GoogleTranslator:
    name = "google"
    URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, cancel=None):
        self.cancel = cancel or threading.Event()
        self.blocked, self.failed = False, 0

    def one(self, text):
        """실패하면 원문을 그대로 돌려준다 (번역 안 된 문단은 영어로 남음)."""
        if self.cancel.is_set():
            raise Cancelled
        if self.blocked:
            self.failed += 1
            return text
        for attempt in range(3):
            try:
                data = urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": text}).encode()
                req = urllib.request.Request(self.URL, data=data, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    if "sorry" in r.geturl():
                        raise PermissionError
                    d = json.loads(r.read().decode("utf-8"))
                return "".join(seg[0] for seg in d[0] if seg and seg[0])
            except (PermissionError, urllib.error.HTTPError) as e:
                if isinstance(e, PermissionError) or e.code in (403, 429):
                    self.failed += 1
                    if not self.blocked:
                        self.blocked = True
                        print("\n  ! 구글이 요청을 막았습니다(너무 많이 보냄). 남은 문단은 영어로 둡니다. "
                              "잠시 뒤 다시 돌리거나 --engine claude 를 쓰세요.")
                    return text
                time.sleep(1 + 2 * attempt)
            except Exception:  # 일시 오류 → 잠깐 쉬고 다시
                time.sleep(1 + 2 * attempt)
        self.failed += 1
        return text

    def run(self, texts, done):
        with cf.ThreadPoolExecutor(6) as ex:
            futs = {ex.submit(self.one, t): t for t in texts}
            for f in cf.as_completed(futs):
                done({futs[f]: f.result()})


class CliError(RuntimeError):
    """claude 명령 자체가 실패 (로그인·한도·실행 오류)."""


class ClaudeTranslator:
    """claude CLI(-p)를 부른다. API 키 변수를 지워서 구독 로그인으로만 돈다."""

    def __init__(self, exe, model, context, token=None, cancel=None):
        self.exe, self.model, self.context = exe, model, context
        self.name = f"claude-{model}"
        self.failed, self.cli_fail, self.fatal = 0, 0, False
        self.lock = threading.Lock()
        self.cancel = cancel or threading.Event()
        self.cwd = tempfile.mkdtemp(prefix="paper_ko_")
        self.env = claude_env(token)
        self.prompt_file = Path(self.cwd) / "system_prompt.txt"
        self.prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")

    def call(self, texts):
        payload = json.dumps({"paper": self.context, "items": texts}, ensure_ascii=False)
        cmd = [self.exe, "-p", "--model", self.model, "--tools", "", "--setting-sources", "",
               "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
               "--system-prompt-file", str(self.prompt_file)]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", env=self.env, cwd=self.cwd,
                                    creationflags=NO_WINDOW)
        except OSError as e:
            raise CliError(str(e)[:300])
        deadline, inp = time.time() + 900, payload
        while True:  # 0.5초마다 취소 확인
            if self.cancel.is_set() or time.time() > deadline:
                proc.kill()
                proc.communicate()
                if self.cancel.is_set():
                    raise Cancelled
                raise CliError("claude 응답 시간 초과 (15분)")
            try:
                stdout, stderr = proc.communicate(inp, timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                inp = None
        try:
            out = json.loads(stdout)
        except ValueError:
            raise CliError((stderr or stdout or "claude 실행 실패").strip()[:300])
        if out.get("is_error"):
            raise CliError(str(out.get("result"))[:300])
        res = out["result"]
        arr = json.loads(res[res.index("["): res.rindex("]") + 1])
        if len(arr) != len(texts):
            raise ValueError(f"개수 불일치 {len(arr)} != {len(texts)}")
        return [str(x) for x in arr]

    def batch(self, texts):
        """CLI 자체가 안 되면(로그인 풀림·사용량 한도) 3번 만에 멈춘다 — 구독 사용량을 헛되이 쓰지 않게.
        답 모양이 틀린 경우만 한 번 더, 그래도 틀리면 반으로 나눠 다시."""
        err = None
        for attempt in range(2):
            if self.cancel.is_set():
                raise Cancelled
            if self.fatal:
                break
            try:
                return dict(zip(texts, self.call(texts)))
            except Cancelled:
                raise
            except CliError as e:
                err = e
                with self.lock:
                    self.cli_fail += 1
                    if self.cli_fail >= 3 and not self.fatal:
                        self.fatal = True
                        print(f"\n  ! claude 실행이 계속 실패해 번역을 멈춥니다: {e}\n"
                              "    (터미널에서 claude 로그인 상태·사용량 한도를 확인하세요)")
                break
            except Exception as e:  # JSON 모양 틀림·개수 불일치
                err = e
        if len(texts) > 1 and not self.fatal and not isinstance(err, CliError):
            mid = len(texts) // 2
            return {**self.batch(texts[:mid]), **self.batch(texts[mid:])}
        with self.lock:
            self.failed += len(texts)
        if not self.fatal:
            print(f"\n  ! 번역 실패, 원문 유지: {texts[0][:50]}... ({err})")
        return {t: t for t in texts}

    def run(self, texts, done):
        batches, cur, n = [], [], 0
        for t in texts:
            if cur and (n + len(t) > 4500 or len(cur) >= 30):
                batches.append(cur)
                cur, n = [], 0
            cur.append(t)
            n += len(t)
        if cur:
            batches.append(cur)
        with cf.ThreadPoolExecutor(4) as ex:
            for f in cf.as_completed([ex.submit(self.batch, b) for b in batches]):
                done(f.result())


class Cache:
    def __init__(self, name):
        self.path = DATA_DIR / "cache" / f"{name}.json"
        try:
            self.d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.d = {}

    @staticmethod
    def key(t):
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def get(self, t):
        return self.d.get(self.key(t))

    def put(self, mapping):
        for t, ko in mapping.items():
            if ko != t:
                self.d[self.key(t)] = ko

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def translate_all(texts, tr, cache, progress=None):
    result, todo = {}, []
    for t in dict.fromkeys(texts):
        hit = cache.get(t)
        if hit is not None:
            result[t] = hit
        else:
            todo.append(t)
    if not todo:
        print(f"  전부 저장된 번역 사용 ({len(result)}개)")
        return result
    total, cnt = len(todo), [0]
    print(f"  {tr.name} 로 {total}개 번역 (저장된 것 {len(result)}개)")
    t0 = time.time()

    def done(m):
        result.update(m)
        cache.put(m)
        cnt[0] += len(m)
        cache.save()
        print(f"\r  번역 {cnt[0]}/{total}  ({time.time() - t0:.0f}초)", end="", flush=True)
        if progress:
            progress(cnt[0] / total)

    tr.run(todo, done)
    print()
    if tr.failed >= len(todo):  # 보낸 것이 전부 실패 = 로그인·토큰·차단 문제 → 영어 그대로인 파일로 덮어쓰지 않는다
        raise RuntimeError("번역을 하나도 못 했습니다 (위 오류 참고). 결과 파일은 만들지 않았습니다 — "
                           "로그인·토큰을 확인하고 다시 돌리세요.")
    if tr.failed:
        print(f"  번역 못 한 문단 {tr.failed}개는 영어로 남겼습니다. 다시 돌리면 그것만 다시 번역합니다.")
    return result


# ───────────────────────────── 그리기 ─────────────────────────────

def fill_maths(ko, maths):
    used = set()

    def rep(m):
        i = int(m.group(1))
        if 1 <= i <= len(maths):
            used.add(i)
            return f"<span class='m'>{maths[i - 1]}</span>"
        return ""

    out = PH.sub(rep, H.escape(ko, quote=False))
    missing = [f"<span class='m'>{maths[i - 1]}</span>" for i in range(1, len(maths) + 1) if i not in used]
    if missing:
        out += " " + " ".join(missing)
    return out


def render(page, paras, arch, debug=False):
    todo = [p for p in paras if p.translate and p.ko]
    if not todo:
        return 0
    # 1) 원문 글자 지우기 (그림·선은 남김). 위아래 줄을 건드리지 않게 가운데 띠만 지정
    for p in todo:
        for r in p.erase:
            dy = r.height * 0.3
            page.add_redact_annot(pymupdf.Rect(r.x0, r.y0 + dy, r.x1, r.y1 - dy), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    # 2) 문단 안 수식에 딸린 선(루트 윗줄, 분수 막대)도 지우기
    math_paras = [p for p in todo if p.maths]
    if math_paras:
        for p in math_paras:
            page.add_redact_annot(p.tight, fill=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                              text=pymupdf.PDF_REDACT_TEXT_NONE)
    # 3) 한국어 넣기 (상자에 안 들어가면 글자를 줄여서 맞춤)
    misfit = 0
    for p in todo:
        body = fill_maths(p.ko, p.maths)
        if p.lead:
            body = f"<b>{H.escape(p.lead_ko or p.lead, quote=False)}</b> {body}"
        fs = max(p.size, 2.0) * 0.94
        lh = max(p.pitch / fs, 1.15)
        style = (f"font-family:{'kog' if p.sans else 'kos'};font-size:{fs:.2f}px;line-height:{lh:.3f};"
                 f"text-align:{p.align};color:#{p.color:06x};" + ("font-weight:bold;" if p.bold else ""))
        html = f'<p style="{style}">{body}</p>'
        spare, scale = page.insert_htmlbox(p.rect, html, css=CSS, archive=arch, scale_low=0.4)
        if spare < 0:
            # 0.4배로도 안 들어가면 아무것도 안 그려진다 → 제한 없이 줄여서라도 넣는다 (원문은 이미 지워짐)
            misfit += 1
            page.insert_htmlbox(p.rect, html, css=CSS, archive=arch, scale_low=0)
    if debug:
        for p in paras:
            page.draw_rect(p.rect, color=(0, 0.7, 0) if p.translate else (1, 0, 0), width=0.4)
            if not p.translate:
                page.insert_text(p.rect.tl + (0, -1), p.why, fontsize=5, color=(1, 0, 0),
                                 fontname="ko", fontfile=str(FONT_DIR / SANS))
    return misfit


# ───────────────────────────── 실행 ─────────────────────────────

def parse_pages(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d*)\s*(-?)\s*(\d*)", part)
        if not part or not m or not (m.group(1) or m.group(3)):
            raise ValueError(f"--pages 형식이 이상합니다: '{part}' (예: 1-3,5  -3  10-)")
        lo = int(m.group(1)) if m.group(1) else 1
        hi = int(m.group(3)) if m.group(3) else (n if m.group(2) else lo)
        out += [i - 1 for i in range(lo, hi + 1) if 1 <= i <= n]
    if not out:
        raise ValueError(f"--pages {spec}: 이 PDF 는 {n}쪽입니다")
    return sorted(set(out))


def save_pdf(doc, path):
    for i in range(20):
        p = path if i == 0 else path.with_name(f"{path.stem}({i}){path.suffix}")
        try:
            doc.save(p, garbage=3, deflate=True)
            return p
        except Exception:  # 같은 이름 파일이 PDF 뷰어에 열려 있으면 이름을 바꿔 저장
            continue
    raise RuntimeError(f"저장 실패: {path}")


def make_translator(args, context):
    cancel = getattr(args, "cancel", None)
    if args.engine == "claude":
        exe = getattr(args, "claude_exe", None) or find_claude()
        if exe:
            # 인증 순서: 창에 넣은 토큰 → .env 토큰 → 환경변수 → 이 PC 로그인
            token = getattr(args, "token", None)
            if token:
                print("  Claude 인증: 창에 넣은 토큰")
            elif env_file_token():
                token = env_file_token()
                print(f"  Claude 인증: .env 토큰 ({env_file_path()})")
            elif os.environ.get(ENV_KEY):
                print(f"  Claude 인증: 환경변수 {ENV_KEY}")
            else:
                print("  Claude 인증: 이 PC 로그인")
            return ClaudeTranslator(exe, args.model, context, token, cancel)
        print("  Claude Code(claude.exe)를 찾지 못해 구글 번역으로 바꿉니다.")
    return GoogleTranslator(cancel)


def translate_pdf(path, args):
    """번역본·나란히 PDF 경로를 (mono, dual) 로 돌려준다. 번역할 게 없으면 None."""
    prog = getattr(args, "progress", None) or (lambda frac, text: None)
    cancel = getattr(args, "cancel", None) or threading.Event()
    path = Path(path).resolve()
    print(f"\n▶ {path.name}")
    prog(0.0, "문단 찾는 중")
    src = pymupdf.open(path)
    pages = parse_pages(args.pages, len(src))
    dicts = {pno: src[pno].get_text("dict", flags=FLAGS) for pno in pages}
    fi = FontInfo(dicts)

    state, by_page = State(), {}
    for pno in pages:
        paras = extract_paras(dicts[pno], pno, fi)
        decide(src[pno], paras, fi, state)
        if pno == 0:
            skip_front_matter(paras)
        by_page[pno] = paras
    allp = [p for ps in by_page.values() for p in ps]
    todo = [p for p in allp if p.translate]
    print(f"  {len(pages)}쪽, 문단 {len(allp)}개 중 번역 {len(todo)}개 (본문 글꼴 {fi.body_font} {fi.body_size}pt)")
    if not todo:
        print("  번역할 영어 문단이 없습니다. (스캔 PDF 라면 글자가 그림이라 이 방식으로는 안 됩니다)")
        return None

    first = by_page[pages[0]]
    title = max(first, key=lambda p: p.size).plain if first else ""
    abstract = next((p.plain for p in allp if p.nrows >= 3), "")
    context = f"제목: {title}\n앞부분: {abstract[:800]}"
    tr = make_translator(args, context)
    texts = [p.src for p in todo if p.src] + [p.lead for p in todo if p.lead]
    prog(0.05, "번역 중")
    mapping = translate_all(texts, tr, Cache(tr.name), lambda f: prog(0.05 + 0.8 * f, "번역 중"))
    if cancel.is_set():
        raise Cancelled
    for p in todo:
        p.ko = re.sub(r"\s*\n\s*", " ", mapping.get(p.src, p.src)) if p.src else ""
        p.lead_ko = mapping.get(p.lead, p.lead) if p.lead else ""
        if not p.src and p.lead:  # 굵은 머리만 있는 문단
            p.ko, p.lead = p.lead_ko, ""

    print("  PDF 에 한국어 넣는 중...")
    out = pymupdf.open(path)
    arch = pymupdf.Archive(str(FONT_DIR))
    misfit = 0
    for k, pno in enumerate(pages):
        if cancel.is_set():
            raise Cancelled
        prog(0.85 + 0.12 * k / len(pages), "PDF 에 한국어 넣는 중")
        misfit += render(out[pno], by_page[pno], arch, args.debug)
    prog(0.97, "저장 중")
    if len(pages) < len(out):
        out.select(pages)
        src.select(pages)
    out.subset_fonts()
    outdir = Path(args.out) if args.out else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    mono = save_pdf(out, outdir / f"{path.stem}_ko.pdf")
    print(f"  ✔ 번역본: {mono}")
    if misfit:
        print(f"  (번역이 길어 글자를 0.4배 아래로 줄인 문단 {misfit}개 — 확대해서 보세요)")
    d = None
    if not args.no_dual:
        out = pymupdf.open(mono)
        dual = pymupdf.open()
        for i in range(len(out)):
            r = out[i].rect
            pg = dual.new_page(width=r.width * 2, height=r.height)
            pg.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), src, i)
            pg.show_pdf_page(pymupdf.Rect(r.width, 0, r.width * 2, r.height), out, i)
        d = save_pdf(dual, outdir / f"{path.stem}_ko_dual.pdf")
        print(f"  ✔ 원문|번역 나란히: {d}")
    prog(1.0, "완료")
    return mono, d


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="논문 PDF 를 글자 자리 그대로 한국어로 번역")
    ap.add_argument("pdf", nargs="*", help="번역할 PDF (여러 개 가능)")
    ap.add_argument("--engine", choices=["claude", "google"], default="claude",
                    help="claude: 구독 로그인 사용, 품질 좋음(기본) / google: 무료, 빠름")
    ap.add_argument("--model", default="sonnet", help="claude 모델 (sonnet, opus, haiku)")
    ap.add_argument("--pages", help="번역할 쪽, 예: 1-3,5")
    ap.add_argument("--out", help="결과 폴더 (기본: 원본 PDF 옆)")
    ap.add_argument("--no-dual", action="store_true", help="원문|번역 나란히 PDF 만들지 않기")
    ap.add_argument("--debug", action="store_true", help="문단 상자와 건너뛴 이유를 표시")
    args = ap.parse_args()
    if not args.pdf:
        print("PDF 파일을 '논문번역.bat' 아이콘 위로 끌어다 놓으세요. 여러 개도 됩니다.\n")
        ap.print_help()
        return
    for f in args.pdf:
        try:
            translate_pdf(f, args)
        except Exception as e:
            print(f"  ✖ 실패: {f}: {e}")
            if args.debug:
                raise


if __name__ == "__main__":
    main()
