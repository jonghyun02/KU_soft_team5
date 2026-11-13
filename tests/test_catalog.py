"""catalog: 제목 풀기, 문서 묶기, 압축 해제, 읽기·갱신, 실제 catalog.csv 수락 기준(A1~A3)."""
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from sinkprep import catalog, detect, paths, util


# ---------------------------------------------------------------- 제목·단서

@pytest.mark.parametrize("title,cat,ft,orig", [
    ("16 Blocks (식스틴 블럭)", "foreign_film", "식스틴 블럭", "16 Blocks"),
    ("미션 임파서블2 Mission: Impossible2", "foreign_film", "미션 임파서블2", "Mission: Impossible2"),
    ("대부Godfather", "foreign_film", "대부", "Godfather"),
    ("gravity 그래비티 pdf", "foreign_film", "그래비티", "gravity"),
    ("L.A. 컨피던셜  L.A. Confidential", "foreign_film", "L.A. 컨피던셜", "L.A. Confidential"),
    ("이더널 션샤인(Eternal Sunshine of the Spotless Mind) shooting script입니다", "foreign_film",
     "이더널 션샤인", "Eternal Sunshine of the Spotless Mind"),
    ("참을 수 없는 사랑 (원제: Intolerable Cruelty),2003", "foreign_film", "참을 수 없는 사랑", "Intolerable Cruelty"),
    ("Pan's Labyrinth (판의 미로 / 스페인어버전)", "foreign_film", "판의 미로", "Pan's Labyrinth"),
    ("타이타닉 <한글판>", "foreign_film", "타이타닉", ""),
    ("<박쥐 Thirst> 영문 시나리오", "ko_film", "박쥐", ""),
    ("<공동경비구역 JSA> 시나리오", "ko_film", "공동경비구역 JSA", ""),
    ("영화\"광해\" 시나리오 입니다", "ko_film", "광해", ""),
    ("이창동 감독 '시' 시나리오", "ko_film", "시", ""),
    ("[요청 자료] 오래된 정원", "ko_film", "오래된 정원", ""),
    ("[번지점프를 하다]의 심의대본입니다.", "ko_film", "번지점프를 하다", ""),
    ("주유소 습격사건 심의대본", "ko_film", "주유소 습격사건", ""),
    ("달콤한 인생(김지운)", "ko_film", "달콤한 인생", ""),
    ("오!수정 시나리오", "ko_film", "오!수정", ""),
])
def test_parse_title(title, cat, ft, orig):
    pt = catalog.parse_title(title, cat)
    assert (pt["film_title"], pt["film_title_orig"]) == (ft, orig)


def test_year_from_title():
    assert catalog.parse_title("라스베가스를 떠나며 (Leaving Las Vegas),1995", "foreign_film")["year"] == "1995"
    assert catalog.parse_title("Crash (크래쉬) 2004년작", "foreign_film")["year"] == "2004"
    assert catalog.parse_title("첫사랑(1993)", "ko_film") == {
        "title_norm": "첫사랑", "film_title": "첫사랑", "film_title_orig": "", "year": "1993"}


@pytest.mark.parametrize("texts,label", [
    (["교회누나-최종고2(제본용)"], "bound"),
    (["eternal sunshine_first draft"], "draft"),
    (["GoneGirl_Final_Shooting_Script"], "shooting"),
    (["봄날은_간다_5고_(2월21일)"], "revised"),
    (["애니홀 한글 번역"], "translation"),
    (["인터뷰_심의대본"], "unknown"),
    (["오!수정 시나리오"], "unknown"),
])
def test_version_cues(texts, label):
    assert catalog.version_cues(texts)[0] == label


def test_lang_and_translation_cues():
    assert catalog.lang_cue(["블레이드 러너 Blade Runner (English/Korean)"]) == "mixed"
    assert catalog.lang_cue(["추락의 해부 대본 (영어)"]) == "en"
    assert catalog.lang_cue(["Pan's Labyrinth (판의 미로 / 스페인어버전)"]) == "es"
    assert catalog.lang_cue(["터미네이터(한글번역판)"]) == "ko"
    assert catalog.lang_cue(["색즉시공 한글로 된거..."]) is None      # HWP(한글) 파일이라는 뜻
    assert catalog.translation_cue(["가위손(한)"]) and not catalog.translation_cue(["가위손(영)"])


def test_norm_key():
    assert catalog.norm_key("The Dark Knight") == catalog.norm_key("Dark Knight") == "darkknight"
    assert catalog.norm_key("와니와 준하") == catalog.norm_key("와니와준하")
    assert catalog.norm_key("Green Mile, The") == "greenmile"


# ---------------------------------------------------------------- 문서 묶기(가짜 사례)

def _entry(tmp_path, name, data=b"", fmt="hwp5", post="100", origin="raw", order=0):
    p = tmp_path / f"{post}_{order}_{Path(name).name}"
    p.write_bytes(data or (name.encode("utf-8") + b"\x00" * 16))
    return catalog.Entry(rel=f"x/{p.name}", path=p, post_id=post, category="ko_film", title="제목",
                         name=name, order=(order, 0), fmt=fmt, sha1=util.sha1_file(p),
                         size=p.stat().st_size, origin=origin)


def test_parts_merge_into_one_doc(tmp_path):
    es = [_entry(tmp_path, "감시자들_2.pdf", fmt="pdf_scan", order=2),
          _entry(tmp_path, "감시자들_1.pdf", fmt="pdf_scan", order=1)]
    docs = catalog.group_post("100", es, [])
    assert len(docs) == 1 and docs[0].doc_id == "fm100"
    assert [f.name for f in docs[0].files] == ["감시자들_1.pdf", "감시자들_2.pdf"]


@pytest.mark.parametrize("names", [
    ["Blade Runner.hwp", "블레이드 러너(영한).hwp"],
    ["달콤한 인생.hwp", "달콤한 인생97.hwp"],
    ["Apocalypse Now(ds).hwp", "Apocalypse Now(ts).hwp"],
    ["eternalsunshine_shootingscript.pdf", "eternal sunshine_first draft.pdf"],
])
def test_different_scripts_get_k(tmp_path, names):
    es = [_entry(tmp_path, n, order=i) for i, n in enumerate(names, start=1)]
    docs = catalog.group_post("100", es, [])
    assert [d.doc_id for d in docs] == ["fm100_1", "fm100_2"]
    assert all(d.status == "pending" and len(d.files) == 1 for d in docs)


def test_part_numbers_must_be_consecutive():
    assert catalog.part_info("황해.part2.rar") == ("황해", 2)
    assert catalog.part_info("시나리오 1부.hwp") == ("시나리오", 1)
    assert catalog.part_info("대본(하).hwp") == ("대본", 3)


def test_excluded_rows_exe_subtitle_duplicate_archive(tmp_path):
    same = b"HWP Document File V3.00 same bytes" + b"\x00" * 40
    a = _entry(tmp_path, "본편.hwp", data=same, fmt="hwp3", order=1)
    dup = _entry(tmp_path, "본편 사본.hwp", data=same, fmt="hwp3", order=2)
    exe = _entry(tmp_path, "MovieSmiCh.exe", data=b"MZ\x90\x00" + b"\x00" * 60, fmt="other", order=3)
    smi = _entry(tmp_path, "영화.smi", data=b"<SAMI><BODY>", fmt="txt", order=4)
    arc = _entry(tmp_path, "묶음.zip", data=b"PK\x03\x04" + b"\x00" * 30, fmt="zip", order=5)
    docs = catalog.group_post("100", [a, dup, exe, smi], [([arc], "archive_container:extracted/100")])
    by_id = {d.doc_id: d for d in docs}
    assert by_id["fm100"].status == "pending" and by_id["fm100"].files == [a]
    notes = {d.doc_id: (d.status, d.note) for d in docs if d.doc_id != "fm100"}
    # 제외 행은 문서 뒤에 파일 순서대로, 압축 행은 맨 뒤
    assert notes == {
        "fm100_2": ("excluded", ["duplicate_file_of:fm100"]),
        "fm100_3": ("excluded", ["not_script:exe"]),
        "fm100_4": ("excluded", ["not_script:subtitle"]),
        "fm100_5": ("excluded", ["archive_container:extracted/100"]),
    }


def test_volume_sets_across_posts_and_parts(tmp_path):
    a = _entry(tmp_path, "LMS.alz", fmt="alz", post="56868")
    b = _entry(tmp_path, "LMS.a00", fmt="alz", post="56874")
    r1 = _entry(tmp_path, "황해.part1.rar", fmt="rar", post="773767", order=1)
    r2 = _entry(tmp_path, "황해.part2.rar", fmt="rar", post="773767", order=2)
    z1 = _entry(tmp_path, "a.zip", fmt="zip", post="1")
    z2 = _entry(tmp_path, "a.zip", fmt="zip", post="2")
    for e in (a, b, r1, r2, z1, z2):
        e.category = "foreign_film"
    sets = catalog.plan_archives([b, a, r2, r1, z1, z2])
    got = {s.owner: [v.name for v in s.volumes] for s in sets}
    assert got == {"1": ["a.zip"], "2": ["a.zip"], "56868": ["LMS.alz", "LMS.a00"],
                   "773767": ["황해.part1.rar", "황해.part2.rar"]}


def _row(doc_id, film_title, fmt="hwp5", size=100, tr="false", status="pending", version="unknown"):
    r = {k: "" for k in catalog.FIELDS}
    r.update({"doc_id": doc_id, "post_id": re.match(r"fm(\d+)", doc_id).group(1), "category": "foreign_film",
              "film_title": film_title, "film_title_orig": "", "title_norm": film_title, "sha1": doc_id,
              "is_translation": tr, "status": status, "is_primary": "false", "note": "",
              "version_label": version, "_rank": catalog._rank(fmt), "_size": size,
              "_container": False, "_not_script": False})
    return r


def test_films_groups_and_primary():
    rows = [_row("fm10", "가위손", "hwp3", 900, tr="true"), _row("fm11", "가위손", "hwp3", 100),
            _row("fm12", "가 위 손!", "pdf_text", 50), _row("fm13", "다른 영화", "txt"),
            _row("fm14", "이터널 선샤인", "pdf_text", 250, version="draft"),
            _row("fm15", "이터널 선샤인", "pdf_text", 240, version="shooting")]
    catalog.assign_films(rows)
    by = {r["doc_id"]: r for r in rows}
    assert by["fm10"]["film_id"] == by["fm11"]["film_id"] == by["fm12"]["film_id"] == "local_0001"
    assert by["fm13"]["film_id"] == "local_0002" and by["fm13"]["dup_group"] == ""
    assert by["fm10"]["dup_group"] == "G001"
    # pdf_text 가 형식 우선순위 1위, 번역 대본은 뒤로
    assert [r["doc_id"] for r in rows if r["is_primary"] == "true"] == ["fm12", "fm13", "fm15"]
    assert catalog.check_primary(rows) == []


# ---------------------------------------------------------------- 압축 해제(반디집)

@pytest.mark.skipif(not paths.BANDIZIP.exists(), reason="반디집 없음")
def test_bandizip_extracts_korean_names(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    z = src / "묶음.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("커밍아웃-감독version.hwp", b"HWP Document File V3.00 ")
        zf.writestr("폴더/대본 2.txt", "가나다".encode("cp949"))
    out = tmp_path / "out"
    out.mkdir()
    ok, log = catalog._bz(z, out)
    assert ok, log
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert got == ["커밍아웃-감독version.hwp", "폴더/대본 2.txt"]
    assert catalog._check_tree(out) == []


# ---------------------------------------------------------------- 읽기·갱신

def test_load_finalize_roundtrip(tmp_path):
    rows = [_row("fm1", "가"), _row("fm2", "가"), _row("fm3", "나")]
    for r in rows:
        r["files"] = f"a/{r['doc_id']}.hwp|a/{r['doc_id']}_2.hwp"
        r["file_formats"] = "hwp5|hwp3"
        r["real_format"] = "mixed"
    catalog.assign_films(rows)
    p = tmp_path / "catalog.csv"
    catalog.write_catalog(rows, p)
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
    got = catalog.load(p)
    assert [r["doc_id"] for r in got] == ["fm1", "fm2", "fm3"]
    assert got[0]["files_list"] == ["a/fm1.hwp", "a/fm1_2.hwp"]
    assert got[0]["file_formats_list"] == ["hwp5", "hwp3"]
    catalog.finalize({"fm3": {"script_lang": "ko", "extract_quality": 0.91, "is_translation": False}}, path=p)
    again = {r["doc_id"]: r for r in catalog.load(p)}
    assert (again["fm3"]["script_lang"], again["fm3"]["extract_quality"], again["fm3"]["is_translation"]) == \
        ("ko", "0.91", "false")
    with pytest.raises(ValueError):          # A2 를 어기면 쓰지 않는다
        catalog.finalize({"fm1": {"is_primary": True}, "fm2": {"is_primary": True}}, path=p)
    with pytest.raises(KeyError):
        catalog.finalize({"fm1": {"no_such_column": 1}}, path=p)
    assert sum(r["is_primary"] == "true" for r in catalog.load(p)) == 2


# ---------------------------------------------------------------- 실제 catalog.csv(있을 때만)

REAL = paths.CATALOG.exists() and (paths.RAW / "manifest.csv").exists()
real = pytest.mark.skipif(not REAL, reason="catalog.csv 나 원본 없음(먼저 python -I -m sinkprep catalog)")


@pytest.fixture(scope="module")
def rows():
    return catalog.load()


@real
def test_header_and_ids(rows):
    with open(paths.CATALOG, encoding="utf-8-sig", newline="") as f:
        assert next(csv.reader(f)) == catalog.FIELDS
    ids = [r["doc_id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"fm\d+(_\d+)?", i) for i in ids)
    assert all(r["doc_id"].startswith("fm" + r["post_id"]) for r in rows)
    # catalog 단계는 pending/excluded 만 쓰고, review 가 parsed 등으로 갱신한다(실제 catalog.csv 는 review 뒤일 수 있음)
    assert {r["status"] for r in rows} <= {"pending", "parsed", "aligned", "done", "excluded"}
    assert all(r["note"] for r in rows if r["status"] == "excluded")
    assert all(r["category"] in ("ko_film", "foreign_film") for r in rows)


@real
def test_A1_every_file_exactly_once(rows):
    on_disk = [paths.rel(p) for p in catalog.raw_files()]
    on_disk += [paths.rel(p) for p in paths.EXTRACTED.rglob("*") if p.is_file()] if paths.EXTRACTED.exists() else []
    listed = Counter(f for r in rows for f in r["files_list"])
    assert [f for f, n in listed.items() if n > 1] == []
    assert set(listed) == set(on_disk)
    # manifest 의 받은 첨부 수 = 디스크 원본 + 디스크에 없는 첨부(note 에 표시)
    ok = [m for m in catalog.read_manifest() if m["status"] == "ok"]
    missing = [r for r in rows if "missing_attachment:" in r["note"]]
    assert len(ok) == len(catalog.raw_files()) + len(missing)


@real
def test_A2_one_primary_per_film(rows):
    assert catalog.check_primary(rows) == []
    live_films = {r["film_id"] for r in rows if r["status"] != "excluded"}
    prim = Counter(r["film_id"] for r in rows if r["is_primary"] == "true")
    assert set(prim) == live_films and set(prim.values()) == {1}
    assert all(r["status"] != "excluded" for r in rows if r["is_primary"] == "true")
    for g in {r["dup_group"] for r in rows if r["dup_group"]}:
        members = [r for r in rows if r["dup_group"] == g]
        assert len(members) >= 2 and len({r["film_id"] for r in members}) == 1


@real
def test_A3_formats_match_bytes(rows):
    files = [(f, fmt) for r in rows for f, fmt in zip(r["files_list"], r["file_formats_list"])]
    assert all(r["real_format"] == (r["file_formats_list"][0] if len(set(r["file_formats_list"])) == 1 else "mixed")
               for r in rows)
    bad = [(f, fmt, detect.catalog_format(detect.real_format(paths.absp(f))))
           for f, fmt in files if detect.catalog_format(detect.real_format(paths.absp(f))) != fmt]
    assert bad == []
    raw_fmt = Counter(fmt for f, fmt in files if f.startswith("filmmakers_scripts/"))
    assert raw_fmt == Counter({"hwp5": 289, "hwp3": 137, "pdf_text": 112, "pdf_scan": 72, "doc": 57, "txt": 15,
                               "archive": 12, "rtf": 2, "docx": 2, "xls": 1, "pdf_broken": 1, "other": 1})


@real
def test_known_judgments(rows):
    by = {r["doc_id"]: r for r in rows}
    assert by["fm5691757"]["files_list"] == [
        "filmmakers_scripts/한국영화 시나리오/감시자들 [5691757] 1_감시자들_1.pdf",
        "filmmakers_scripts/한국영화 시나리오/감시자들 [5691757] 2_감시자들_2.pdf"]
    assert [by[f"fm54259_{k}"]["status"] == "excluded" for k in range(1, 6)] == [False] * 4 + [True]
    assert by["fm773767"]["files_list"] == ["extracted/773767/황해.pdf"]
    assert by["fm56868"]["files_list"] == ["extracted/56868/LITTLE_MISS_SUNSHINE.pdf"]
    assert by["fm56874"]["note"] == "archive_container:extracted/56868"
    assert by["fm5691397_2"]["note"] == "duplicate_file_of:fm5691397"
    assert by["fm55922"]["note"] == "duplicate_file_of:fm55536"
    assert by["fm54331"]["note"] == "not_script:other"
    assert (by["fm56791"]["film_title"], by["fm56791"]["film_title_orig"]) == ("식스틴 블럭", "16 Blocks")
    assert by["fm55542"]["real_format"] == "rtf" and by["fm56704"]["real_format"] == "txt"
    assert by["fm56178"]["film_id"] != by["fm56181"]["film_id"]          # 미션 임파서블 1·2 는 다른 영화
    assert by["fm56910"]["film_id"] == by["fm56924"]["film_id"]          # Dark Knight = The Dark Knight


def test_carry_over_keeps_later_columns(tmp_path):
    """catalog 를 다시 만들어도 원본이 그대로인 문서는 review·영상 칸을 잃지 않는다. 원본이 바뀐 문서는 새 값."""
    from sinkprep import catalog as C

    def row(doc, film, sha, **kw):
        r = {k: "" for k in C.FIELDS}
        r.update(doc_id=doc, film_id=film, files=f"filmmakers_scripts/x/{doc}.pdf", sha1=sha, status="pending",
                 script_lang="unknown", is_primary="false", is_translation="false")
        r.update(kw)
        return r

    old = [row("fm1", "local_1", "a", status="parsed", script_lang="ko", is_primary="true", extract_quality="0.97",
               video_path="videos/f1.mp4", match="0.91"),
           row("fm2", "local_1", "b", status="parsed", script_lang="en"),
           row("fm3", "local_2", "c", status="parsed", script_lang="ko", is_primary="true")]
    p = tmp_path / "catalog.csv"
    C.write_catalog(old, p)
    new = [row("fm1", "local_1", "a", is_primary="false"), row("fm2", "local_1", "b", is_primary="true"),
           row("fm3", "local_2", "CHANGED", is_primary="true")]
    assert C.carry_over(new, p) == 2
    by = {r["doc_id"]: r for r in new}
    assert by["fm1"]["script_lang"] == "ko" and by["fm1"]["video_path"] == "videos/f1.mp4" and by["fm1"]["match"] == "0.91"
    assert (by["fm1"]["is_primary"], by["fm2"]["is_primary"]) == ("true", "false")      # 예전 값이 서로 맞음
    assert by["fm3"]["script_lang"] == "unknown" and by["fm3"]["status"] == "pending"      # 원본이 바뀜
    # 이어받은 true(fm1) 와 원본이 바뀐 문서의 새 임시값 true(fm2) 가 겹치면 그 영화는 새 임시값으로
    new = [row("fm1", "local_1", "a", is_primary="false"), row("fm2", "local_1", "B2", is_primary="true")]
    assert C.carry_over(new, p) == 1
    assert C.check_primary(new) == [] and [r["is_primary"] for r in new] == ["false", "true"]


def test_unsafe_archive_names():
    """풀기 전에 바깥으로 나가는 항목 이름(.., 절대 경로, 드라이브)을 막는다(zip-slip)."""
    from sinkprep import catalog as C
    bs = chr(92)
    assert [C._unsafe_name(n) for n in ["a/b.txt", "x..y/z", "황해.pdf"]] == [False] * 3
    assert all(C._unsafe_name(n) for n in ["../x", ".." + bs + "x", "C:/w", "/etc", "a/../../b", bs + "x"])
