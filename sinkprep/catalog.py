"""catalog 단계: 원본 목록 -> 압축 해제 -> 형식 판별 -> 문서 묶기 -> catalog.csv

- 원본(paths.RAW)은 읽기만 한다. 압축은 반디집 CLI 로 extracted/{post_id}/ 새 빈 폴더에 푼다.
  분할 압축은 볼륨 이름이 깨져 있어 manifest 의 원래 첨부 이름으로 cache/stage/{post_id}/ 에 복사한 뒤 푼다.
- 글 하나 = 문서 하나가 기본. 한 대본의 조각(_1/_2, part1/2, 1부/2부)은 문서 하나에 files 여럿,
  서로 다른 대본은 fm{글번호}_{k}, 대본이 아닌 파일·압축 자체·같은 파일 중복은 status=excluded + note 사유.
- catalog 단계는 원본만 보고 정할 수 있는 칸을 채운다. script_lang·is_translation·is_primary 는 임시값이고
  review 단계가 finalize() 로 고친다.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import stat
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import detect, paths, util

STAGE = "catalog"
VERSION = "0.1.0"

FIELDS = [
    "doc_id", "post_id", "category", "files", "sha1", "real_format", "file_formats",
    "title_raw", "title_norm", "script_lang", "is_translation", "version_label", "version_raw",
    "dup_group", "film_id", "film_title", "film_title_orig", "year",
    "video_path", "video_sha1", "video_duration", "video_lang",
    "is_primary", "match", "extract_quality", "status", "note",
]
CATEGORY = {"한국영화 시나리오": "ko_film", "외국영화 시나리오": "foreign_film"}
# is_primary 임시값: 형식 우선순위(앞일수록 좋음), 같은 형식이면 판본(영화에 가까운 쪽), 그다음 파일 크기
FORMAT_RANK = ["pdf_text", "hwp5", "docx", "doc", "rtf", "hwp3", "txt", "pdf_scan"]
VERSION_PREF = ["shooting", "final", "bound", "revised", "unknown", "draft", "translation"]
SUBTITLE_EXT = {".smi", ".sami", ".srt", ".sub", ".ass", ".ssa", ".vtt"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
EXE_EXT = {".exe", ".com", ".dll", ".msi", ".bat", ".cmd", ".scr"}
DOC_FORMATS = set(detect.SCRIPT_FORMATS) | {"xls"}   # 문서(대본)로 받는 형식

# ------------------------------------------------------------------ 손 판단(하나하나 확인한 글)
# 제목 규칙으로 못 푸는 글. 값은 원본 파일·제목을 직접 보고 정했다(보고서 '판단 목록').
POST_OVERRIDES: dict[str, dict] = {
    "765450": {"film_title": "아저씨"},                       # 오랜만에 올립니다.. 아저씨..
    "54980": {"film_title": "말아톤"},                        # 말아톤 - 상영이 끝나서 올립니다..
    "56854": {"film_title": "여인의 향기"},                   # 알파치노.. 여인의 향기 입니다..
    "56046": {"film_title": "나인 야드", "film_title_orig": "The Whole Nine Yards"},  # 첨부 이름
    "56847": {"film_title": "완행열차"},                      # ...칸영화제 단편부문 최우수작품
    "21201869": {"film_title": "플리백", "film_title_orig": "Fleabag", "year": "2016",
                 "note": "tv_episode:S1E1"},
    "4067697": {"film_title": "", "film_title_orig": "Tangerine", "year": "2015"},  # - Sean Baker
    "56730": {"film_title": "스파이더맨", "film_title_orig": "Spider Man",
              "version_raw": "제임스 카메론 감독 버전"},
    "56697": {"film_title": "킹콩", "film_title_orig": "King Kong", "version_raw": "피터잭슨 버전"},
    "55346": {"film_title": "지리멸렬", "year": "1994"},      # 봉준호 감독, 지리멸렬 (Incoherence, 支離滅裂, 1994)
    "54806": {"film_title": "지리멸렬"},                      # 지리멸렬-봉준호 감독 단편
    "54335": {"film_title": "후아유"},                        # 'whoru" (첨부 후아유.hwp)
    "54331": {"film_title": "반칙왕"},                        # 드디어 반칙왕 시나라오를.....
    "55932": {"film_title": "오버 더 레인보우"},              # <오보더 레인보우> (제목 오타)
    "56878": {"film_title": "와일드 혹스", "film_title_orig": "Wild Hogs",
              "note": "content:subtitle_transcript"},        # 한글·영문 자막 시각 목록
    "55474": {"note": "content:subtitle_transcript"},        # [빈집] 자막추출
    "54448": {"note": "content:synopsis"},                   # 일진녀는 못말려 시놉시스
    "54537": {"note": "content:recording_script_xls"},       # 녹음대본(엑셀)
    "4141439": {"note": "tv_episode:10"},                    # 드라마 [태양의 후예] 10회
    "56700": {"note": "tv_series"},
}
# (글 번호, 원래 첨부 이름) -> 칸 덮어쓰기
FILE_OVERRIDES: dict[tuple[str, str], dict] = {
    # 내용 확인: ds = 1975 초기 판본("early version of the screenplay"), ts = 영화 받아쓰기(Transcript)
    ("56560", "Apocalypse Now(ds).hwp"): {"version_label": "draft", "version_raw": "ds"},
    ("56560", "Apocalypse Now(ts).hwp"): {"version_label": "unknown", "version_raw": "ts",
                                          "note": "content:transcript"},
}

# ------------------------------------------------------------------ 제목 단서
VERSION_CUES: list[tuple[str, str]] = [
    ("shooting", r"촬영\s*(?:대본|본|고)|shooting\s*script|(?<![a-z])shooting(?![a-z])"),
    ("bound", r"제본\s*(?:고|판|용|본)?"),
    ("final", r"최종\s*(?:고|본)?\s*\d*|파이널|(?<![a-z])final(?![a-z])|완고|완성본"),
    ("revised", r"수정\s*(?:고|본|판)|(?<![가-힣!])수정(?![가-힣])|개정\s*(?:고|판)?|(?<![a-z])revised(?![a-z])|진행고"
                r"|(?<![\d.\[\]])(?:[2-9]|\d{2,})(?:\[\d\])?(?:\.\d+)?\s*고(?![가-힣])"),
    ("draft", r"초고|초안|first\s*draft|(?<![a-z])draft(?![a-z])|(?:원작자\s*)?최초\s*버전|(?<![\d.\[\]])1\s*고(?![가-힣])"),
    ("translation", r"한글\s*번역판?|번역\s*(?:판|본)?|한글판"),
    ("unknown", r"심의\s*대본|녹음\s*대본|감독\s*(?:판|버전|version)|작가의\s*[a-z]\s*버전|오리지[널날](?:\s*각본)?"
                r"|원작\s*각본|레이아웃\s*편집본|배포용|ver\.?\s*[\d.]+|가제"),
]
VERSION_PRIORITY = ["shooting", "bound", "final", "revised", "draft", "translation", "unknown"]

LANG_CUES: list[tuple[str, str]] = [
    ("mixed", r"english\s*/\s*korean|korean\s*/\s*english|영한|한영|한글\s*\+\s*영문|영문\s*\+\s*한글"),
    ("es", r"스페인어|(?<![a-z])spanish"),
    ("en", r"\(\s*영어\s*\)|영문판?|\(\s*영\s*\)|(?<![a-z])english(?![a-z])"),
    ("ko", r"한글\s*번역판?|한글판|\(\s*한\s*\)|번역\s*(?:판|본)?"),
]
TRANSLATION_CUE = r"번역|한글판|\(\s*한\s*\)"

JUNK = [
    r"\[\s*요청\s*자료\s*\]", r"\[\s*시나리오\s*\]", r"요청\s*자료", r"요청하신",
    r"다시\s*올립니다", r"올립니다", r"원문도", r"원문", r"원제\s*:",
    r"시나리오\s*공모\s*당선작",
    r"시나리오\s*입니다", r"시나리옵니다", r"시나리오", r"시나료", r"씨날", r"시나라오를?", r"시놉시스",
    r"대본", r"스크립트", r"screenplay", r"(?<![a-z])script(?![a-z])",
    r"입니다", r"임당", r"입당",
    r"(?<![a-z])pdf(?![a-z])(?:\s*파일)?", r"(?<![a-z])hwp(?![a-z])",
    r"\(\s*영어\s*\)", r"\(\s*영문판?\s*\)", r"영문판?", r"\(\s*english\s*/\s*korean\s*\)",
    r"english\s*/\s*korean", r"한글\s*번역판?", r"한글판", r"번역판", r"번역본", r"번역",
    r"\(\s*한글\s*\+\s*영문\s*\)", r"자막\s*추출", r"팩키지|패키지", r"\(\s*일본(?:\s*애니)?\s*\)",
    r"한글로\s*된거", r"/?\s*스페인어\s*버전", r"(?<![a-z])part\.?\s*\d+", r"\(\s*드라마\s*\)", r"^드라마\s+",
]
_JUNK_RE = [re.compile(p, re.I) for p in JUNK]
_VERSION_RE = [(lab, re.compile(p, re.I)) for lab, p in VERSION_CUES]
_LANG_RE = [(lab, re.compile(p, re.I)) for lab, p in LANG_CUES]
_HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_LATIN = re.compile(r"[A-Za-z]")
_YEAR_RES = [re.compile(r"\(\s*((?:19|20)\d\d)\s*\)"), re.compile(r",\s*((?:19|20)\d\d)\b"),
             re.compile(r"((?:19|20)\d\d)\s*년작")]
_BRACKETS = [re.compile(r"<+([^<>]+)>+"), re.compile(r"《([^》]+)》"), re.compile(r"\"([^\"]+)\""),
             re.compile(r"“([^”]+)”"), re.compile(r"'([^']*[가-힣][^']*)'"), re.compile(r"\[([^\]]+)\]")]


def _nfc(s: str | None) -> str:
    return unicodedata.normalize("NFC", s or "")


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_edges(s: str) -> str:
    s = re.sub(r"\(\s*\)|<\s*>|\[\s*\]", "", s)
    s = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))
    s = _ws(s)
    s = re.sub(r"^[\s\-–:/,·~]+", "", s)
    s = re.sub(r"[\s\-–:/,·~.]+$", "", s)
    return _ws(s)


def _remove_junk(s: str) -> str:
    for _, rx in _VERSION_RE:      # 판본 표시('심의대본')를 먼저, 그다음 군더더기('대본')
        s = rx.sub(" ", s)
    for rx in _JUNK_RE:
        s = rx.sub(" ", s)
    return _strip_edges(s)


def has_hangul(s: str) -> bool:
    return bool(_HANGUL.search(s or ""))


def has_latin(s: str) -> bool:
    return bool(_LATIN.search(s or ""))


def norm_key(s: str | None) -> str:
    """같은 영화 판단용 열쇠: NFKC, 소문자, 앞 the·뒤 ', the' 제거, 글자·숫자만."""
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"^the\s+|,\s*the$", "", s)
    return "".join(ch for ch in s if ch.isalnum())


def version_cues(texts: list[str]) -> tuple[str, str]:
    """(version_label, version_raw). texts 는 제목·첨부 이름(확장자 뺌)."""
    found: list[tuple[str, str]] = []
    for t in texts:
        t = _nfc(t)
        hits = []
        for lab, rx in _VERSION_RE:
            for m in rx.finditer(t):
                hits.append((m.start(), lab, _ws(m.group(0))))
        for _, lab, raw in sorted(hits):
            if raw and all(raw != r for _, r in found):
                found.append((lab, raw))
    if not found:
        return "unknown", ""
    labs = {lab for lab, _ in found}
    label = next(lab for lab in VERSION_PRIORITY if lab in labs)
    return label, " ".join(r for _, r in found)


def lang_cue(texts: list[str]) -> str | None:
    for t in texts:
        for lab, rx in _LANG_RE:
            if rx.search(_nfc(t)):
                return lab
    return None


def translation_cue(texts: list[str]) -> bool:
    return any(re.search(TRANSLATION_CUE, _nfc(t)) for t in texts)


def extract_year(texts: list[str]) -> str:
    for t in texts:
        for rx in _YEAR_RES:
            m = rx.search(_nfc(t))
            if m:
                return m.group(1)
    return ""


def split_titles(t: str) -> tuple[str, str]:
    """외국영화 제목 -> (한글 제목, 원어 제목)."""
    t = _strip_edges(t)
    m = re.match(r"^(?P<a>[^()]*?)\s*[-–]?\s*\((?P<b>[^()]*)\)\s*(?P<rest>.*)$", t)
    if m:
        a, b = _strip_edges(m["a"]), _strip_edges(m["b"])
        if a and b and has_hangul(b) and not has_hangul(a):
            return b, a
        if a and b and has_hangul(a) and not has_hangul(b) and has_latin(b):
            return a, b
    if not has_hangul(t):
        return "", t
    if not has_latin(t):
        return t, ""
    first, end = _hangul_span(t)
    before, after = t[:first], t[end:]
    if has_latin(before) and not has_latin(after):      # 'gravity 그래비티', 'Rush - 러시'
        return _strip_edges(t[first:]), _strip_edges(before)
    return _strip_edges(t[:end]), _strip_edges(after)   # '대부2 Godfather : Part II', 'L.A. 컨피던셜 L.A. Confidential'


def _hangul_span(t: str) -> tuple[int, int]:
    """첫 한글 위치, 마지막 한글 뒤에 붙은 숫자·기호까지 포함한 끝 위치('미션 임파서블2', '되나요?')."""
    idx = [i for i, ch in enumerate(t) if _HANGUL.match(ch)]
    end = idx[-1] + 1
    while end < len(t) and not t[end].isspace() and not _LATIN.match(t[end]):
        end += 1
    return idx[0], end


def parse_title(title_raw: str, category: str) -> dict:
    """글 제목 -> title_norm, film_title, film_title_orig, year(제목만 보고)."""
    t = _ws(_nfc(title_raw))
    year = extract_year([t])
    for rx in _YEAR_RES:
        t = rx.sub(" ", t)
    # <제목>·"제목"·[제목] 처럼 괄호로 감싼 제목이 있으면 가장 앞의 것(군더더기만 든 괄호는 건너뜀)
    cands = []
    for rx in _BRACKETS:
        for m in rx.finditer(t):
            inner = _remove_junk(m.group(1))
            if inner:
                cands.append((m.start(), inner))
                break
    body = min(cands)[1] if cands else _remove_junk(t)
    body = body.strip("'\"“” ")
    if category == "ko_film":
        while True:
            nb = re.sub(r"\s*\([^()]*\)\s*$", "", body)
            if nb == body:
                break
            body = _strip_edges(nb)
        ko, orig = body, ""
        if has_hangul(body) and has_latin(body) and has_hangul(body[:1] or ""):
            # 한글 제목 뒤 영어 제목('박쥐 Thirst', '홍길동의 후예 The Descendants…')만 뗀다. 'JSA' 같은 약어는 둔다.
            _, end = _hangul_span(body)
            if re.search(r"[a-z]", body[end:]):
                ko = _strip_edges(body[:end])
        return {"title_norm": _strip_edges(ko), "film_title": _strip_edges(ko),
                "film_title_orig": "", "year": year}
    ko, orig = split_titles(body)
    return {"title_norm": _strip_edges(body), "film_title": ko, "film_title_orig": orig, "year": year}


# ------------------------------------------------------------------ 원본 목록
@dataclass
class Entry:
    rel: str                    # ROOT 기준 상대 경로(슬래시)
    path: Path
    post_id: str
    category: str               # ko_film | foreign_film
    title: str                  # manifest 글 제목
    name: str                   # 원래 이름(manifest 첨부 이름, 압축 안 파일은 그 이름)
    order: tuple
    fmt: str = ""
    sha1: str = ""
    size: int = 0
    origin: str = "raw"         # raw | extracted
    container: str = ""         # 풀린 파일이면 그 압축이 풀린 폴더(rel)
    nested_dir: str = ""        # 이 파일이 압축이면 풀린 폴더(rel)


def read_manifest(raw: Path | None = None) -> list[dict]:
    p = (raw or paths.RAW) / "manifest.csv"
    with open(p, encoding="utf-8-sig", newline="") as f:
        return [{k: _nfc(v) for k, v in r.items()} for r in csv.DictReader(f)]


def raw_files(raw: Path | None = None) -> list[Path]:
    raw = raw or paths.RAW
    return sorted(p for p in raw.rglob("*") if p.is_file() and not (p.parent == raw and p.name == "manifest.csv"))


def _detect_cached(entries: list[Entry], force: bool) -> None:
    """형식·sha1 계산(크기·수정 시각이 같으면 이전 결과 재사용)."""
    cpath = paths.STATE / STAGE / "detect.json"
    cache = {}
    if cpath.exists():
        try:
            cache = util.read_json(cpath)
        except Exception:
            cache = {}
    new = {}
    for e in entries:
        st = e.path.stat()
        c = None if force else cache.get(e.rel)
        if c and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns and c.get("v") == VERSION:
            e.fmt, e.sha1 = c["fmt"], c["sha1"]
        else:
            e.fmt, e.sha1 = detect.real_format(e.path), util.sha1_file(e.path)
        e.size = st.st_size
        new[e.rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "fmt": e.fmt, "sha1": e.sha1, "v": VERSION}
    merged = {**cache, **new}
    # 사라진 파일 정리
    merged = {k: v for k, v in merged.items() if (paths.ROOT / k).exists()}
    if merged != cache:
        util.write_json(cpath, dict(sorted(merged.items())))


def raw_entries(manifest: list[dict], missing: list[dict]) -> list[Entry]:
    by_saved = {}
    for i, r in enumerate(manifest):
        if r.get("status") == "ok" and r.get("saved_as"):
            by_saved[r["saved_as"]] = (i, r)
    out = []
    seen = set()
    for p in raw_files():
        name = _nfc(p.name)
        hit = by_saved.get(name)
        if hit:
            i, r = hit
            seen.add(name)
            m = re.search(r"\]\s+(\d+)_", name)
            sub = int(m.group(1)) if m else 0
            out.append(Entry(rel=paths.rel(p), path=p, post_id=r["post_id"],
                             category=CATEGORY.get(r["category"], r["category"]), title=r["title"],
                             name=r["attachment"] or name, order=(sub, i)))
        else:  # manifest 에 없는 파일: 이름에서 글 번호를 읽는다
            m = re.search(r"\[(\d+)\]", name)
            pid = m.group(1) if m else "0"
            title = _ws(name.split(" [")[0]) if " [" in name else p.stem
            out.append(Entry(rel=paths.rel(p), path=p, post_id=pid,
                             category=CATEGORY.get(p.parent.name, p.parent.name), title=title,
                             name=name, order=(0, 10 ** 6)))
    for i, r in enumerate(manifest):
        if r.get("status") == "ok" and r.get("saved_as") and r["saved_as"] not in seen:
            missing.append(r)
    return out


# ------------------------------------------------------------------ 압축 해제
_VOL_RES = [
    ("rar", re.compile(r"^(?P<base>.+)\.part(?P<n>\d+)\.rar$", re.I), lambda m: int(m["n"]), True),
    ("alz", re.compile(r"^(?P<base>.+)\.alz$", re.I), lambda m: 0, False),
    ("alz", re.compile(r"^(?P<base>.+)\.a(?P<n>\d\d)$", re.I), lambda m: int(m["n"]) + 1, True),
    ("zip", re.compile(r"^(?P<base>.+)\.z(?P<n>\d\d)$", re.I), lambda m: int(m["n"]), True),
    ("zip", re.compile(r"^(?P<base>.+)\.zip$", re.I), lambda m: 10 ** 4, False),
    ("rar", re.compile(r"^(?P<base>.+)\.r(?P<n>\d\d)$", re.I), lambda m: int(m["n"]) + 1, True),
    ("rar", re.compile(r"^(?P<base>.+)\.rar$", re.I), lambda m: 0, False),
]


def volume_info(name: str) -> tuple[str, str, int, bool]:
    """첨부 원래 이름 -> (계열, 바탕 이름, 볼륨 순서, 분할 표시 여부)."""
    for fam, rx, idx, split in _VOL_RES:
        m = rx.match(name)
        if m:
            return fam, m["base"].lower(), idx(m), split
    return "", name.lower(), 0, False


@dataclass
class ArchiveSet:
    owner: str                  # 풀린 파일이 속할 글 번호(첫 볼륨의 글)
    volumes: list[Entry] = field(default_factory=list)

    @property
    def split(self) -> bool:
        return len(self.volumes) > 1


def plan_archives(entries: list[Entry]) -> list[ArchiveSet]:
    arcs = [e for e in entries if e.fmt in detect.ARCHIVE_FORMATS]
    groups: dict[tuple, list[tuple[int, Entry, bool]]] = defaultdict(list)
    for e in arcs:
        fam, base, idx, split = volume_info(e.name)
        groups[(fam or e.fmt, base, e.category)].append((idx, e, split))
    sets = []
    for key, members in groups.items():
        if len(members) > 1 and not any(s for _, _, s in members):
            # 분할 표시 없는 같은 이름 압축들(서로 다른 글): 따로 푼다
            for idx, e, _ in members:
                sets.append(ArchiveSet(owner=e.post_id, volumes=[e]))
            continue
        members.sort(key=lambda x: (x[0], int(x[1].post_id)))
        sets.append(ArchiveSet(owner=members[0][1].post_id, volumes=[e for _, e, _ in members]))
    sets.sort(key=lambda s: int(s.owner))
    return sets


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "file"


def _bz(archive: Path, outdir: Path) -> tuple[bool, str]:
    """반디집 CLI 로 풀기. -cp:949 로 한글 이름(코드페이지 949)을 맞춘다."""
    paths.assert_not_raw(outdir)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    cmd = [str(paths.BANDIZIP), "x", "-cp:949", "-y", f"-o:{outdir}", str(archive)]
    try:
        r = subprocess.run(cmd, cwd=str(paths.CACHE), stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, f"bz 실행 실패: {ex}"
    raw = (r.stdout or b"") + (r.stderr or b"")
    try:
        out = raw.decode("utf-8")
    except UnicodeDecodeError:
        out = raw.decode("cp949", "replace")
    return r.returncode == 0, f"rc={r.returncode}\n{out.strip()}"


def _is_reparse(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _unsafe_name(name: str) -> bool:
    n = name.replace("\\", "/").strip()
    return (n.startswith("/") or re.match(r"^[A-Za-z]:", n) is not None
            or any(part == ".." for part in n.split("/")))


def _listing_bad(archive: Path) -> list[str]:
    """풀기 전에 반디집 목록(bz l)으로 항목 이름을 보고 바깥으로 나가는 경로(.., 절대 경로, 드라이브)를 찾는다.
    풀린 뒤 검사(_check_tree)는 임시 폴더 안만 보므로, 폴더 밖에 쓰는 항목은 풀기 전에 막아야 한다.
    목록을 못 읽으면 위험으로 본다."""
    cmd = [str(paths.BANDIZIP), "l", "-cp:949", str(archive)]
    try:
        r = subprocess.run(cmd, cwd=str(paths.CACHE), stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return [f"목록을 읽지 못함: {ex}"]
    raw = (r.stdout or b"") + (r.stderr or b"")
    try:
        out = raw.decode("utf-8")
    except UnicodeDecodeError:
        out = raw.decode("cp949", "replace")
    lines = out.splitlines()
    head = next((i for i, ln in enumerate(lines) if ln.rstrip().endswith("Name") or " Name " in ln), None)
    seps = [i for i, ln in enumerate(lines) if ln.startswith("-----")]
    if r.returncode != 0 or head is None or len(seps) < 2:
        return [f"목록을 읽지 못함(rc={r.returncode})"]
    col = lines[head].index("Name")
    return [f"바깥 경로 항목: {ln[col:].strip()}" for ln in lines[seps[0] + 1:seps[1]]
            if _unsafe_name(ln[col:].strip())]


def _check_tree(root: Path) -> list[str]:
    """풀린 폴더 안에 바깥을 가리키는 경로·링크가 없는지 확인."""
    bad = []
    rr = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            if _is_reparse(p):
                bad.append(f"링크: {p}")
            elif rr not in p.resolve().parents:
                bad.append(f"바깥 경로: {p}")
    return bad


def _list_files(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: _nfc(p.relative_to(root).as_posix()))


def _rmtree_extracted(p: Path) -> None:
    rp = p.resolve()
    if paths.EXTRACTED.resolve() not in rp.parents:
        raise PermissionError(f"extracted 밖은 지우지 않습니다: {rp}")
    shutil.rmtree(rp)


def _stage(aset: ArchiveSet) -> Path:
    """분할 압축: 원래 첨부 이름으로 cache/stage/{owner}/ 에 복사(원본은 그대로)."""
    sdir = paths.CACHE / "stage" / aset.owner
    paths.assert_not_raw(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    first = None
    for v in aset.volumes:
        dst = sdir / _safe_name(v.name)
        if not (dst.exists() and dst.stat().st_size == v.size and util.sha1_file(dst) == v.sha1):
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(v.path, tmp)
            os.replace(tmp, dst)
        first = first or dst
    return first


def extract_set(aset: ArchiveSet, force: bool = False) -> dict:
    """압축 묶음 하나를 extracted/{owner}/ 에 푼다(이미 풀렸으면 건너뜀). 결과 지문 dict."""
    target = paths.EXTRACTED / aset.owner
    key = f"extract_{aset.owner}"
    want = {"version": VERSION, "archives": [v.rel for v in aset.volumes], "sha1": [v.sha1 for v in aset.volumes]}
    old = util.stage_fp(STAGE, key)
    if (not force and old and old.get("in") == want and target.is_dir() and old.get("ok")
            and all((paths.ROOT / r).is_file() for r in old.get("files", []))
            and len(_list_files(target)) == len(old.get("files", []))):
        old["skipped"] = True
        return old
    paths.EXTRACTED.mkdir(parents=True, exist_ok=True)
    tmp = paths.EXTRACTED / f".tmp_{aset.owner}_{os.getpid()}"
    if tmp.exists():
        _rmtree_extracted(tmp)
    tmp.mkdir()
    src = _stage(aset) if aset.split else aset.volumes[0].path
    pre = _listing_bad(src)
    if pre:
        _rmtree_extracted(tmp)
        return {"in": want, "ok": False, "files": [], "nested": {}, "problems": ["풀지 않음(위험한 항목): " + "; ".join(pre[:5])],
                "log": [], "skipped": False}
    ok, log = _bz(src, tmp)
    logs = [f"{paths.rel(src) if paths.ROOT in src.resolve().parents else src}: {log}"]
    problems = []
    bad = _check_tree(tmp)
    if bad:
        ok = False
        problems.append("위험한 경로: " + "; ".join(bad[:5]))
    nested = {}
    if ok:
        for f in _list_files(tmp):
            fmt = detect.real_format(f)
            if fmt in detect.ARCHIVE_FORMATS:
                sub = f.with_name(f.name + ".x")
                if sub.exists():
                    problems.append(f"중첩 압축 폴더 이름 충돌: {sub.name}")
                    continue
                sub.mkdir()
                pre2 = _listing_bad(f)
                if pre2:
                    problems.append(f"중첩 압축 풀지 않음(위험한 항목): {f.name}: " + "; ".join(pre2[:3]))
                    continue
                ok2, log2 = _bz(f, sub)
                logs.append(f"(중첩) {f.relative_to(tmp).as_posix()}: {log2}")
                bad2 = _check_tree(sub)
                if not ok2 or bad2:
                    problems.append(f"중첩 압축 해제 실패: {f.name}")
                    if bad2:
                        _rmtree_extracted(sub)
                        sub.mkdir()
                nested[f.relative_to(tmp).as_posix()] = sub.relative_to(tmp).as_posix()
    if not ok:
        problems.insert(0, "압축 해제 실패")
    if target.exists():
        _rmtree_extracted(target)
    os.replace(tmp, target)
    files = [paths.rel(p) for p in _list_files(target)]
    fp = {"in": want, "ok": ok, "files": files, "nested": {f"{paths.rel(target)}/{k}": f"{paths.rel(target)}/{v}"
                                                         for k, v in nested.items()},
          "problems": problems, "log": logs}
    util.set_stage_fp(STAGE, key, fp)
    fp["skipped"] = False
    return fp


# ------------------------------------------------------------------ 문서 묶기
_PART_RES = [
    re.compile(r"^(?P<base>.*?)[\s_\-.]*(?:part|pt)[\s_\-.]*(?P<n>\d{1,2})$", re.I),
    re.compile(r"^(?P<base>.*?)[\s_\-.]*(?P<n>\d{1,2})\s*부$"),
    re.compile(r"^(?P<base>.*?)[\s_\-.]*\(?(?P<n>[상중하])\)?$"),
    re.compile(r"^(?P<base>.*?)\s*\((?P<n>\d{1,2})\)$"),
    re.compile(r"^(?P<base>.*?)[\s_\-]+(?P<n>\d{1,2})$"),
    re.compile(r"^(?P<base>.*?\D)(?P<n>\d{1,2})$"),
]
_PART_KO = {"상": 1, "중": 2, "하": 3}


def part_info(name: str) -> tuple[str, int] | None:
    stem = _nfc(Path(name).stem).strip()
    for rx in _PART_RES:
        m = rx.match(stem)
        if m and m["base"].strip():
            n = m["n"]
            return norm_key(m["base"]), (_PART_KO[n] if n in _PART_KO else int(n))
    return None


def merge_parts(cands: list[Entry]) -> list[list[Entry]]:
    """한 글 안 대본 파일들 -> 문서별 파일 목록. 같은 바탕 이름 + 1(또는 0)부터 이어지는 번호면 한 문서."""
    info = {id(e): part_info(e.name) for e in cands}
    by_base: dict[str, list[Entry]] = defaultdict(list)
    for e in cands:
        pi = info[id(e)]
        if pi:
            by_base[pi[0]].append(e)
    merged: dict[int, list[Entry]] = {}
    for base, es in by_base.items():
        if len(es) < 2:
            continue
        nums = sorted(info[id(e)][1] for e in es)
        if nums[0] in (0, 1) and nums == list(range(nums[0], nums[0] + len(nums))):
            es = sorted(es, key=lambda e: info[id(e)][1])
            for e in es:
                merged[id(e)] = es
    out, used = [], set()
    for e in cands:
        if id(e) in used:
            continue
        grp = merged.get(id(e), [e])
        used.update(id(x) for x in grp)
        out.append(grp)
    return out


def _not_script_kind(e: Entry) -> str | None:
    ext = Path(e.name).suffix.lower() or e.path.suffix.lower()
    if ext in SUBTITLE_EXT:
        return "subtitle"
    if e.fmt in DOC_FORMATS:
        return None
    try:
        with open(e.path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        head = b""
    if head.startswith(b"MZ") or ext in EXE_EXT:
        return "exe"
    if ext in IMAGE_EXT or head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM")):
        return "image"
    return e.fmt or "other"


@dataclass
class Doc:
    post_id: str
    category: str
    title: str
    files: list[Entry]
    status: str = "pending"
    note: list[str] = field(default_factory=list)
    dup_of: Entry | None = None     # 같은 글 안 중복이면 원래 파일
    doc_id: str = ""
    k: int = 0


def group_post(post_id: str, entries: list[Entry], containers: list[tuple[list[Entry], str]]) -> list[Doc]:
    """한 글의 파일 -> 문서 목록(문서 먼저, 제외 행 뒤). doc_id 까지 붙인다."""
    if entries:
        category, title = entries[0].category, entries[0].title
    else:
        category, title = containers[0][0][0].category, containers[0][0][0].title
    raws = sorted([e for e in entries if e.origin == "raw"], key=lambda e: e.order)
    exts = sorted([e for e in entries if e.origin == "extracted"], key=lambda e: _nfc(e.rel))
    docs: list[Doc] = []
    excluded: list[Doc] = []
    cands: list[Entry] = []
    first_by_sha: dict[str, Entry] = {}
    for e in raws + exts:
        kind = _not_script_kind(e)
        if kind:
            excluded.append(Doc(post_id, category, title, [e], "excluded", [f"not_script:{kind}"]))
            continue
        if e.sha1 in first_by_sha:
            excluded.append(Doc(post_id, category, title, [e], "excluded", dup_of=first_by_sha[e.sha1]))
            continue
        first_by_sha[e.sha1] = e
        cands.append(e)
    for grp in merge_parts(cands):
        docs.append(Doc(post_id, category, title, grp))
    for files, note in containers:
        excluded.append(Doc(post_id, category, title, files, "excluded", [note]))
    rows = docs + excluded
    if len(rows) == 1:
        rows[0].doc_id, rows[0].k = f"fm{post_id}", 0
    elif len(docs) == 1:
        docs[0].doc_id, docs[0].k = f"fm{post_id}", 0
        for i, d in enumerate(excluded, start=2):
            d.doc_id, d.k = f"fm{post_id}_{i}", i
    else:
        for i, d in enumerate(rows, start=1):
            d.doc_id, d.k = f"fm{post_id}_{i}", i
    owner = {id(f): d for d in docs for f in d.files}
    for d in excluded:
        if d.dup_of is not None:
            d.note = [f"duplicate_file_of:{owner[id(d.dup_of)].doc_id}"]
    return rows


# ------------------------------------------------------------------ 칸 채우기
def _rank(fmt: str) -> int:
    return FORMAT_RANK.index(fmt) if fmt in FORMAT_RANK else len(FORMAT_RANK)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _doc_sort_key(doc_id: str) -> tuple:
    m = re.match(r"fm(\d+)(?:_(\d+))?$", doc_id)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10 ** 12, 0)


def build_rows(docs: list[Doc], posts_multi: set[str]) -> list[dict]:
    rows = []
    for d in docs:
        names = [_nfc(Path(f.name).stem) for f in d.files]
        po = POST_OVERRIDES.get(d.post_id, {})
        pt = parse_title(d.title, d.category)
        single = d.post_id not in posts_multi
        cue_texts = ([d.title] if single else []) + names
        vlabel, vraw = version_cues(cue_texts)
        lang = lang_cue(cue_texts)
        if lang is None and re.search(r"원문", d.title) and d.category == "foreign_film" \
                and re.fullmatch(r"[\x00-\x7f]+", pt["film_title_orig"] or pt["title_norm"] or "x"):
            lang = "en"
        is_tr = d.category == "foreign_film" and translation_cue(cue_texts) and lang != "mixed"
        year = pt["year"] or (extract_year(names) if d.category == "foreign_film" else "")
        fmts = [detect.catalog_format(f.fmt) for f in d.files]
        row = {k: "" for k in FIELDS}
        row.update({
            "doc_id": d.doc_id, "post_id": d.post_id, "category": d.category,
            "files": "|".join(f.rel for f in d.files), "sha1": "|".join(f.sha1 for f in d.files),
            "real_format": fmts[0] if len(set(fmts)) == 1 else "mixed", "file_formats": "|".join(fmts),
            "title_raw": d.title, "title_norm": pt["title_norm"],
            "script_lang": lang or "unknown", "is_translation": "true" if is_tr else "false",
            "version_label": vlabel, "version_raw": vraw,
            "film_title": pt["film_title"], "film_title_orig": pt["film_title_orig"], "year": year,
            "is_primary": "false", "status": d.status,
        })
        notes = list(d.note)
        for kk in ("title_norm", "film_title", "film_title_orig", "year", "version_label", "version_raw"):
            if kk in po:
                row[kk] = po[kk]
        if "title_norm" not in po and ("film_title" in po or "film_title_orig" in po):
            row["title_norm"] = po.get("film_title") or po.get("film_title_orig") or row["title_norm"]
        if po.get("note") and d.status != "excluded":
            notes.append(po["note"])
        for f in d.files:
            fo = FILE_OVERRIDES.get((d.post_id, _nfc(f.name)))
            if fo:
                for kk, vv in fo.items():
                    if kk == "note":
                        notes.append(vv)
                    else:
                        row[kk] = vv
        if row["version_raw"] and row["version_label"] == "unknown" and "version_label" not in po:
            row["version_label"] = version_cues([row["version_raw"]])[0]
        row["note"] = ";".join(dict.fromkeys(n for n in notes if n))
        row["_size"] = sum(f.size for f in d.files)
        row["_rank"] = min(_rank(f.fmt) for f in d.files)
        row["_container"] = any(n.startswith("archive_") for n in d.note)
        row["_not_script"] = any(n.startswith("not_script") for n in d.note)
        if row["_container"] or row["_not_script"]:   # 대본이 아닌 행에는 영화 칸을 두지 않는다
            row.update({"film_title": "", "film_title_orig": "", "year": "", "script_lang": "unknown",
                        "version_label": "unknown", "version_raw": ""})
        rows.append({k: util.nfc(v) if isinstance(v, str) else v for k, v in row.items()})
    return rows


def assign_films(rows: list[dict]) -> None:
    """film_id·dup_group·영화 칸 통일·is_primary(임시값)."""
    uf = _UF()
    film_rows = [r for r in rows if not r["_container"] and not r["_not_script"]]
    keyed: dict[str, list[str]] = defaultdict(list)
    for r in film_rows:
        did = r["doc_id"]
        uf.find(did)
        cat = r["category"]
        if r["film_title"] and norm_key(r["film_title"]):
            keyed[f"{cat}|ko|{norm_key(r['film_title'])}"].append(did)
        elif not r["film_title_orig"] and norm_key(r["title_norm"]):
            keyed[f"{cat}|ko|{norm_key(r['title_norm'])}"].append(did)
        if r["film_title_orig"] and norm_key(r["film_title_orig"]):
            keyed[f"{cat}|orig|{norm_key(r['film_title_orig'])}"].append(did)
        keyed[f"sha|{r['sha1']}"].append(did)
        m = re.match(r"duplicate_file_of:(\S+)", r["note"])
        if m:
            keyed[f"dup|{m.group(1)}"] += [did, m.group(1)]
    for ids in keyed.values():
        for x in ids[1:]:
            uf.union(ids[0], x)
    films: dict[str, list[dict]] = defaultdict(list)
    for r in film_rows:
        films[uf.find(r["doc_id"])].append(r)
    order = sorted(films.values(), key=lambda rs: min(_doc_sort_key(r["doc_id"]) for r in rs))
    g = 0
    for i, rs in enumerate(order, start=1):
        rs.sort(key=lambda r: _doc_sort_key(r["doc_id"]))
        fid = f"local_{i:04d}"
        title = next((r["film_title"] for r in rs if r["film_title"]), "")
        orig = next((r["film_title_orig"] for r in rs if r["film_title_orig"]), "")
        year = next((r["year"] for r in rs if r["year"]), "")
        live = [r for r in rs if r["status"] != "excluded"]
        grp = ""
        if len(live) >= 2:
            g += 1
            grp = f"G{g:03d}"
        for r in rs:
            r.update({"film_id": fid, "film_title": title, "film_title_orig": orig, "year": year,
                      "dup_group": grp if r["status"] != "excluded" else ""})
        if live:
            best = min(live, key=lambda r: (r["is_translation"] == "true", r["_rank"],
                                            VERSION_PREF.index(r["version_label"])
                                            if r["version_label"] in VERSION_PREF else len(VERSION_PREF),
                                            -r["_size"], _doc_sort_key(r["doc_id"])))
            best["is_primary"] = "true"


# ------------------------------------------------------------------ 실행
def collect(force: bool = False, log=print) -> tuple[list[dict], dict]:
    """원본·압축을 모두 훑어 catalog 행(정렬됨)과 요약을 만든다."""
    manifest = read_manifest()
    missing: list[dict] = []
    entries = raw_entries(manifest, missing)
    _detect_cached(entries, force)
    sets = plan_archives(entries)
    by_post: dict[str, list[Entry]] = defaultdict(list)
    containers: dict[str, list[tuple[list[Entry], str]]] = defaultdict(list)
    arc_ids = {id(v) for s in sets for v in s.volumes}
    for e in entries:
        if id(e) not in arc_ids:
            by_post[e.post_id].append(e)
    problems: dict[str, list[dict]] = defaultdict(list)
    meta = {"extract": {}}
    post_info = {e.post_id: e for e in entries}
    for s in sets:
        res = extract_set(s, force=force)
        meta["extract"][s.owner] = res
        log(f"  압축 {s.owner}: {'건너뜀(이미 풀림)' if res.get('skipped') else '풀기'} "
            f"-> 파일 {len(res['files'])}개{'' if res['ok'] else ' (실패)'}")
        owner = post_info[s.owner]
        tdir = paths.rel(paths.EXTRACTED / s.owner)
        # 압축 행: 글마다 하나(볼륨 여러 개면 같은 행)
        per_post: dict[str, list[Entry]] = defaultdict(list)
        for v in s.volumes:
            per_post[v.post_id].append(v)
        for pid, vols in per_post.items():
            containers[pid].append((sorted(vols, key=lambda v: v.order), f"archive_container:{tdir}"))
        nested = res.get("nested", {})
        ext_entries = []
        for r in res["files"]:
            p = paths.ROOT / r
            cont = next((d for d in nested.values() if r.startswith(d + "/")), tdir)
            ext_entries.append(Entry(rel=r, path=p, post_id=s.owner, category=owner.category, title=owner.title,
                                     name=_nfc(p.name), order=(0, 0), origin="extracted", container=cont))
        _detect_cached(ext_entries, force)
        for e in ext_entries:
            if e.rel in nested:     # 중첩 압축 파일 자체 -> 압축 행
                containers[s.owner].append(([e], f"archive_container:{nested[e.rel]}"))
            elif e.fmt in detect.ARCHIVE_FORMATS:   # 두 단계보다 깊은 압축
                containers[s.owner].append(([e], "archive_nested_not_extracted"))
            else:
                by_post[s.owner].append(e)
        for msg in res.get("problems", []):
            problems[s.owner].append(msg)
    docs: list[Doc] = []
    posts_multi = set()
    for pid in sorted(set(by_post) | set(containers), key=int):
        ds = group_post(pid, by_post.get(pid, []), containers.get(pid, []))
        if sum(1 for d in ds if d.status != "excluded") > 1:
            posts_multi.add(pid)
        docs.extend(ds)
    rows = build_rows(docs, posts_multi)
    rows.sort(key=lambda r: _doc_sort_key(r["doc_id"]))
    # 다른 글과 똑같은 파일(들)이면 뒤 글을 중복으로 제외
    first_by_sha: dict[str, str] = {}
    for r in rows:
        if r["status"] == "excluded":
            continue
        if r["sha1"] in first_by_sha:
            r["status"] = "excluded"
            r["note"] = ";".join([f"duplicate_file_of:{first_by_sha[r['sha1']]}"] + ([r["note"]] if r["note"] else []))
        else:
            first_by_sha[r["sha1"]] = r["doc_id"]
    assign_films(rows)
    # 빠진 첨부(manifest 에는 ok 인데 디스크에 없음)
    for m in missing:
        pid = m["post_id"]
        target = next((r for r in rows if r["post_id"] == pid and r["status"] != "excluded"), None)
        if target is not None:
            target["note"] = ";".join(x for x in [target["note"], f"missing_attachment:{m['attachment']}"] if x)
    meta["missing"] = missing
    meta["problems"] = problems
    return rows, meta


def _problem_list(rows: list[dict], meta: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)

    def add(key, level, problem, detail, row=None, film_id="", score=None, path=""):
        out[key].append({"level": level, "doc_id": row["doc_id"] if row else "", "film_id": film_id,
                         "item_id": "", "problem": problem, "score": score, "detail": detail, "path": path})

    for r in rows:
        if r["status"] == "excluded":
            continue
        for f, fmt in zip(r["files"].split("|"), r["file_formats"].split("|")):
            if fmt == "pdf_broken":
                add(r["doc_id"], "doc", "extract_broken", "PDF 를 열거나 글자를 꺼낼 수 없음(catalog 판별)",
                    r, r["film_id"], 0.0, f)
        if "missing_attachment:" in r["note"]:
            add(r["doc_id"], "doc", "other", "manifest 에는 받은 첨부인데 디스크에 없음: "
                + r["note"].split("missing_attachment:", 1)[1].split(";")[0], r, r["film_id"])
    for pid, msgs in meta.get("problems", {}).items():
        r = next((x for x in rows if x["post_id"] == pid and "archive_container" in x["note"]), None)
        key = r["doc_id"] if r else f"fm{pid}"
        for msg in msgs:
            add(key, "doc", "other", msg, r, "", None, r["files"].split("|")[0] if r else "")
    # 영화 ID 를 KMDb/TMDB 에서 찾지 못함(키 없음) -> local ID, 확인 필요
    seen = set()
    for r in rows:
        fid = r["film_id"]
        if fid.startswith("local_") and fid not in seen and r["status"] != "excluded":
            seen.add(fid)
            out[fid].append({"level": "film", "doc_id": "", "film_id": fid, "item_id": "", "problem": "other",
                             "score": None,
                             "detail": f"KMDb/TMDB 조회 불가(키 없음): local ID, 정규화 제목 "
                                       f"'{r['film_title'] or r['film_title_orig']}'",
                             "path": ""})
    return out


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: util.nfc(str(r.get(k, "") if r.get(k) is not None else "")) for k in FIELDS})
    return buf.getvalue()


def write_catalog(rows: list[dict], path: Path | None = None) -> None:
    util.write_text(Path(path) if path else paths.CATALOG, to_csv(rows), encoding="utf-8-sig")


def summary(rows: list[dict]) -> dict:
    live = [r for r in rows if r["status"] != "excluded"]
    exc = [r for r in rows if r["status"] == "excluded"]
    files = [f for r in rows for f in r["files"].split("|") if f]
    fmts = Counter(fmt for r in rows for fmt in r["file_formats"].split("|") if fmt)
    films = {r["film_id"] for r in live}
    groups = {r["dup_group"] for r in live if r["dup_group"]}
    return {
        "rows": len(rows), "docs": len(live), "excluded": len(exc), "files": len(files),
        "raw_files": sum(1 for f in files if f.startswith(paths.rel(paths.RAW) + "/")),
        "extracted_files": sum(1 for f in files if f.startswith(paths.rel(paths.EXTRACTED) + "/")),
        "file_formats": dict(sorted(fmts.items(), key=lambda x: -x[1])),
        "doc_real_format": dict(Counter(r["real_format"] for r in live).most_common()),
        "excluded_notes": dict(Counter(r["note"].split(":")[0] for r in exc).most_common()),
        "films": len(films), "dup_groups": len(groups),
        "dup_group_docs": sum(1 for r in live if r["dup_group"]),
        "multi_file_docs": sum(1 for r in live if "|" in r["files"]),
        "primary": sum(1 for r in live if r["is_primary"] == "true"),
    }


def run(args) -> int:
    force = bool(getattr(args, "force", False))
    if getattr(args, "docs", None) or getattr(args, "limit", None):
        print("catalog 는 전체 원본을 한꺼번에 묶어야 해서 --docs/--limit 를 무시합니다.")
    rows, meta = collect(force=force)
    kept = carry_over(rows)
    if kept:
        print(f"catalog: 원본이 그대로인 문서 {kept}개는 뒤 단계가 채운 칸(review·영상)을 이어받음")
    write_catalog(rows)
    probs = _problem_list(rows, meta)
    pdir = paths.PROBLEMS / STAGE
    old_keys = {p.stem for p in pdir.glob("*.json")} if pdir.exists() else set()
    for key in sorted(old_keys - set(probs)):
        util.set_problems(STAGE, key, [])
    for key, ps in probs.items():
        util.set_problems(STAGE, key, ps)
    s = summary(rows)
    util.write_json(paths.STATE / STAGE / "summary.json", s)
    print(f"catalog.csv: 행 {s['rows']} (문서 {s['docs']}, 제외 {s['excluded']}), 파일 {s['files']} "
          f"(원본 {s['raw_files']}, 압축 해제 {s['extracted_files']})")
    print(f"  파일 형식: {s['file_formats']}")
    print(f"  문서 형식: {s['doc_real_format']}")
    print(f"  제외 사유: {s['excluded_notes']}")
    print(f"  영화 {s['films']}편, 같은 영화 묶음 {s['dup_groups']}개(문서 {s['dup_group_docs']}), "
          f"여러 파일 문서 {s['multi_file_docs']}, is_primary {s['primary']}")
    if meta["missing"]:
        print(f"  디스크에 없는 첨부: {[m['saved_as'] for m in meta['missing']]}")
    return 0


# ------------------------------------------------------------------ 읽기·갱신
def load(path: Path | str | None = None) -> list[dict]:
    """catalog.csv -> 행 dict 목록(files_list, file_formats_list, sha1_list 추가)."""
    p = Path(path) if path else paths.CATALOG
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["files_list"] = [x for x in r.get("files", "").split("|") if x]
        r["file_formats_list"] = [x for x in r.get("file_formats", "").split("|") if x]
        r["sha1_list"] = [x for x in r.get("sha1", "").split("|") if x]
    return rows


# 뒤 단계가 채우는 칸: review(글자를 읽고 정함)와 영상 정렬(영상 파일·일치율). catalog 를 다시 만들 때
# 원본(files·sha1)이 그대로인 문서는 이 칸을 지우지 않고 이어받는다.
LATER_COLUMNS = ("script_lang", "is_translation", "extract_quality", "status", "is_primary",
                 "video_path", "video_sha1", "video_duration", "video_lang", "match")


def carry_over(rows: list[dict], path: Path | str | None = None) -> int:
    """지금 catalog.csv 에서 같은 doc_id·files·sha1 인 행의 LATER_COLUMNS 값을 새 행에 옮긴다(제외 행은 그대로).
    옮긴 뒤 영화마다 is_primary 가 둘 이상이면(영화 묶음이 바뀜) 그 영화는 새로 정한 임시값으로 되돌린다. 옮긴 행 수."""
    p = Path(path) if path else paths.CATALOG
    if not p.exists():
        return 0
    try:
        old = {r["doc_id"]: r for r in load(p)}
    except Exception:  # noqa: BLE001 - 예전 파일이 깨졌으면 새로 만든 값을 쓴다
        return 0
    fresh = {r["doc_id"]: {c: r.get(c, "") for c in LATER_COLUMNS} for r in rows}
    n = 0
    for r in rows:
        o = old.get(r["doc_id"])
        if not o or r["status"] == "excluded" or o.get("status") == "excluded" \
                or o.get("files") != r["files"] or o.get("sha1") != r["sha1"] or o.get("film_id") != r["film_id"]:
            continue
        for c in LATER_COLUMNS:
            if o.get(c, "") != "":
                r[c] = o[c]
        n += 1
    bad = set(check_primary(rows))
    for r in rows:
        if r["film_id"] in bad:
            r["is_primary"] = fresh[r["doc_id"]]["is_primary"]
    return n


def check_primary(rows: list[dict]) -> list[str]:
    """A2: film_id 마다 is_primary=true 는 최대 1개. 어긴 film_id 목록."""
    c = Counter(r["film_id"] for r in rows if r.get("is_primary") == "true" and r.get("film_id"))
    return sorted(f for f, n in c.items() if n > 1)


def finalize(updates: dict[str, dict] | None = None, rows: list[dict] | None = None,
             path: Path | str | None = None) -> list[dict]:
    """review 단계용: 글자를 읽고 정한 칸(script_lang, is_translation, extract_quality, is_primary, status, match,
    note 등)을 catalog.csv 에 반영하고 원자적으로 다시 쓴다.

    updates: {doc_id: {칸: 값}}. rows 를 주면 그 행들을 그대로 쓴다(load() 결과를 고친 것).
    film_id 마다 is_primary=true 가 둘 이상이면 ValueError.
    """
    rows = rows if rows is not None else load(path)
    if updates:
        by_id = {r["doc_id"]: r for r in rows}
        for did, upd in updates.items():
            if did not in by_id:
                raise KeyError(f"catalog 에 없는 doc_id: {did}")
            for k, v in upd.items():
                if k not in FIELDS:
                    raise KeyError(f"catalog 칸이 아님: {k}")
                if isinstance(v, bool):
                    v = "true" if v else "false"
                by_id[did][k] = "" if v is None else str(v)
    bad = check_primary(rows)
    if bad:
        raise ValueError(f"is_primary 가 둘 이상인 film_id: {bad[:10]}")
    for r in rows:
        r["files_list"] = [x for x in r.get("files", "").split("|") if x]
        r["file_formats_list"] = [x for x in r.get("file_formats", "").split("|") if x]
        r["sha1_list"] = [x for x in r.get("sha1", "").split("|") if x]
    write_catalog(rows, Path(path) if path else None)
    return rows
