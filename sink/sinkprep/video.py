"""영상 등록(video): 영상 파일 하나를 한 영화에 잇고 catalog 의 영상 칸을 채운다.

python -I -m sinkprep video <영상 경로> --docs <doc_id> [--lang ko] [--force]
python -I -m sinkprep video --unregister --docs <doc_id> [--force]

하는 일
1. 그 문서의 film_id 를 찾는다.
2. 영상 sha1 과 길이(컨테이너 길이, 소수 3자리)를 구한다. 영상 줄이 없는 파일(소리만)은 받지 않는다.
3. 영상을 videos/{film_id}{확장자} 로 하드링크한다(안 되면 복사). 원본 영상은 고치거나 옮기지 않는다.
   catalog.csv 는 git 에 들어가므로 video_path 에는 이 상대 경로(ROOT 기준, 슬래시)만 적는다.
4. 소리를 꺼내고(cache/audio/{sha1}.wav) 소리 시각 ↔ 영상 시각 대응표를 만든다.
5. 4 가 끝난 뒤에 그 영화의 제외되지 않은 모든 문서 행에 video_path, video_sha1, video_duration, video_lang 을
   한 번에 쓴다(소리를 못 꺼낸 영상이 catalog 에 남지 않게).

--lang 이 없으면 그 영화에 이미 적힌 video_lang 을 그대로 두고, 그것도 없으면 한국영화 게시판(ko_film) 문서는 ko,
아니면 unknown. 지역 꼬리가 붙은 값(ko-KR, en-US)은 기본 언어(ko, en)로 줄여 적는다: 뒤 단계(final)가 줄 언어를
기본 언어로 견주므로 ko-KR 을 그대로 두면 대본에 없는 말 전부가 한국어가 아닌 줄로 잡힌다.
이미 같은 값이고 새로 만든 것이 없으면 skip.
그 영화에 다른 영상(sha1 이 다름)이 이미 등록돼 있으면 확장자와 상관없이 --force 가 있어야 바꾼다. 바꾸면
videos/ 에 남은 그 영화의 옛 파일을 지운다(원본은 건드리지 않는다. 옛 영상의 cache 는 남는다).

등록 해제(--unregister): 그 영화의 모든 문서 행에서 영상 칸 넷과 match 를 비우고, status 가 aligned·done 이면 parsed
로 되돌리고, 그 영화의 asr·align·final 문제 기록을 지운다. videos/{film_id}.* 는 다른 하드링크가 있는 파일(원본이
따로 있음)만 지우고, 링크가 하나뿐인 파일(링크를 못 해 복사했거나 원본 자체)은 --force 일 때만 지운다.
films/{film_id}/ 의 결과와 캐시(cache/audio, cache/asr, cache/state)는 지우지 않고 남은 것을 알린다.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import media, paths

STAGE = "video"
LANG_RE = re.compile(r"^([a-z]{2,3}(-[A-Za-z0-9]{2,8})?|mixed|unknown)$")
COLUMNS = ("video_path", "video_sha1", "video_duration", "video_lang")
FILM_STAGES = ("asr", "align", "final")          # 등록 해제 때 문제 기록을 지우는 영화 단계
USAGE = ("쓰는 법: python -I -m sinkprep video <영상 경로> --docs <doc_id> [--lang ko]\n"
         "        python -I -m sinkprep video --unregister --docs <doc_id> [--force]")


def base_lang(lang: str) -> tuple[str, str]:
    """(적을 언어 값, 줄인 까닭 또는 빈 글). 'ko-KR' -> ('ko', 까닭). 꼴이 틀리면 ValueError."""
    if not LANG_RE.match(lang):
        raise ValueError(f"--lang 값이 언어 코드 꼴이 아님: {lang!r} (ko, en, ja, mixed, unknown ...)")
    base = lang.split("-")[0]
    if base == lang:
        return lang, ""
    return base, (f"언어 {lang!r} 는 기본 언어 {base!r} 로 줄여 적음(final 이 줄 언어를 기본 언어로 견주므로 "
                  "지역 꼬리를 남기면 대본에 없는 말이 모두 외국어 줄로 잡힘)")


def link_video(src: Path, film_id: str, sha1: str, *, force: bool = False) -> tuple[Path, str]:
    """영상을 videos/{film_id}{확장자} 에 둔다. (경로, 한 일): link(하드링크), copy(복사), same(이미 같은 파일),
    kept(이미 같은 내용의 파일). 같은 이름의 다른 영상이 있으면 force 일 때만 바꾼다."""
    dst = paths.VIDEOS / f"{film_id}{src.suffix.lower()}"
    if dst.exists():
        if os.path.samefile(src, dst):
            media.remember_sha1(dst, sha1)
            return dst, "same"
        if media.file_sha1(dst) == sha1:
            return dst, "kept"
        if not force:
            raise media.MediaError(f"videos/{dst.name} 에 다른 영상이 이미 있음(바꾸려면 --force)")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        part = dst.with_name(dst.name + ".part")
        try:
            shutil.copy2(src, part)
            os.replace(part, dst)
        finally:
            if part.exists():
                part.unlink()
        if media.file_sha1(dst) != sha1:
            dst.unlink()
            raise media.MediaError(f"복사한 영상의 sha1 이 원본과 다름: {dst}")
        return dst, "copy"
    media.remember_sha1(dst, sha1)
    return dst, "link"


def stale_videos(film_id: str, keep: Path) -> list[Path]:
    """videos/ 에 있는 그 영화의 다른 파일(확장자가 다른 옛 등록). keep 과 쓰다 만 .part 는 뺀다."""
    if not paths.VIDEOS.is_dir():
        return []
    return sorted(p for p in paths.VIDEOS.glob(f"{film_id}.*")
                  if p.is_file() and p != keep and p.suffix.lower() != ".part")


def gap_note(tm: media.TimeMap) -> str:
    if tm.mode == "padded":
        return "패킷 표로 시각을 이을 수 없어 틈을 무음으로 메워 뽑음(소리 시각 = 영상 시각)"
    if tm.gaps:
        return f"영상 시각 틈 {tm.gaps}곳 합 {tm.gap_total:.3f}초: 소리 시각을 영상 시각으로 옮겨 씀"
    if tm.mode == "map":
        return f"소리가 영상 시각 {tm.segments[0][1]:.3f}초에서 시작: 소리 시각을 영상 시각으로 옮겨 씀"
    return "영상 시각 틈 없음(소리 시각 = 영상 시각)"


def register(video, doc_id: str, lang: str | None = None, *, force: bool = False) -> tuple[str, str]:
    """영상을 doc_id 의 영화에 등록한다. ("ok"|"skip", 설명)."""
    from . import catalog

    rows = catalog.load()
    row = next((r for r in rows if r["doc_id"] == doc_id), None)
    if row is None:
        raise KeyError(f"catalog 에 없는 doc_id: {doc_id}")
    if row.get("status") == "excluded":
        raise ValueError(f"제외된 문서: {doc_id} ({row.get('note', '')})")
    film_id = row.get("film_id") or ""
    if not film_id:
        raise ValueError(f"film_id 가 없는 문서: {doc_id}")
    # --lang 이 없으면 앞서 등록할 때 준 언어를 지킨다(다시 등록했다고 ko·unknown 으로 돌아가지 않게)
    lang = lang or (row.get("video_lang") or "").strip() or ("ko" if row.get("category") == "ko_film" else "unknown")
    lang, lang_note = base_lang(lang)
    src = Path(video)
    if not src.is_file():
        raise FileNotFoundError(f"영상 파일이 없음: {video}")

    info = media.probe(src)
    if info["video"] is None:
        raise media.MediaError(f"영상 줄이 없는 파일(소리만 있음): {video}")
    if info["audio"] is None:
        raise media.MediaError(f"소리 스트림이 없는 영상: {video}")
    if not info["duration"] or info["duration"] <= 0:
        raise media.MediaError(f"영상 길이를 알 수 없음: {video}")
    sha1 = media.file_sha1(src)
    old_sha = (row.get("video_sha1") or "").strip()
    if old_sha and old_sha != sha1 and not force:
        raise media.MediaError(f"{film_id} 에 다른 영상({row.get('video_path')}, sha1 {old_sha[:8]})이 이미 등록돼 있음"
                               "(바꾸려면 --force)")
    dst, how = link_video(src, film_id, sha1, force=force)

    # 소리와 대응표를 먼저 만든다. 여기서 실패하면 catalog 는 그대로다
    made = not media.audio_path(sha1).exists() or not media.timemap_path(sha1).exists()
    wav = media.extract_audio(dst, sha1)
    tm = media.build_timemap(dst, sha1, wav)

    want = {"video_path": f"videos/{dst.name}", "video_sha1": sha1,
            "video_duration": f"{info['duration']:.3f}", "video_lang": lang}
    updates = {r["doc_id"]: dict(want) for r in rows
               if r.get("film_id") == film_id and r.get("status") != "excluded"
               and any((r.get(c) or "") != want[c] for c in COLUMNS)}
    if updates:
        catalog.finalize(updates=updates)
    removed = []
    for p in stale_videos(film_id, dst):                     # 확장자가 다른 옛 등록 파일
        try:
            p.unlink()
            removed.append(p.name)
        except OSError as e:
            print(f"  ! videos/{p.name} 을 지울 수 없음: {e}", file=sys.stderr, flush=True)
    how_ko = {"link": "하드링크", "copy": "복사", "same": "이미 등록된 파일", "kept": "이미 같은 영상"}[how]
    msg = (f"{film_id} <- {want['video_path']} ({how_ko}), sha1 {sha1[:8]}, {want['video_duration']}초, "
           f"언어 {lang}, catalog {len(updates)}행 갱신 | {gap_note(tm)}")
    if removed:
        msg += f" | 옛 파일 지움: {', '.join(removed)}"
    if lang_note:
        msg += f" | {lang_note}"
    return ("ok" if updates or how in ("link", "copy") or made or removed else "skip"), msg


def unregister(doc_id: str, *, force: bool = False) -> tuple[str, str]:
    """doc_id 의 영화에서 영상 등록을 지운다(맨 위 설명의 '등록 해제'). ("ok"|"skip", 설명)."""
    from . import catalog, util

    rows = catalog.load()
    row = next((r for r in rows if r["doc_id"] == doc_id), None)
    if row is None:
        raise KeyError(f"catalog 에 없는 doc_id: {doc_id}")
    film_id = row.get("film_id") or ""
    if not film_id:
        raise ValueError(f"film_id 가 없는 문서: {doc_id}")
    sha1s = {r["video_sha1"] for r in rows if r.get("film_id") == film_id and r.get("video_sha1")}
    updates = {}
    for r in rows:
        if r.get("film_id") != film_id:
            continue
        want = {c: "" for c in COLUMNS + ("match",)}
        if r.get("status") in ("aligned", "done"):
            want["status"] = "parsed"
        if any((r.get(c) or "") != v for c, v in want.items()):
            updates[r["doc_id"]] = want
    if updates:
        catalog.finalize(updates=updates)
    cleared = []
    for stage in FILM_STAGES:
        if (paths.PROBLEMS / stage / f"{film_id}.json").exists():
            util.set_problems(stage, film_id, [])
            cleared.append(stage)
    removed, kept = [], []
    for p in sorted(paths.VIDEOS.glob(f"{film_id}.*")) if paths.VIDEOS.is_dir() else []:
        if not p.is_file():
            continue
        if p.stat().st_nlink > 1 or force or p.suffix.lower() == ".part":
            try:
                p.unlink()
                removed.append(p.name)
            except OSError as e:
                print(f"  ! videos/{p.name} 을 지울 수 없음: {e}", file=sys.stderr, flush=True)
                kept.append(p.name)
        else:
            kept.append(p.name)
    left = []
    fdir = paths.FILMS / film_id
    if fdir.is_dir():
        n = sum(1 for p in fdir.rglob("*") if p.is_file())
        left.append(f"films/{film_id}/ 파일 {n}개")
    for sha in sorted(sha1s):
        for p in (media.audio_path(sha), media.timemap_path(sha)):
            if p.exists():
                left.append(p.relative_to(paths.ROOT).as_posix() if p.is_relative_to(paths.ROOT) else str(p))
    for stage in FILM_STAGES:
        if util.stage_fp(stage, film_id) is not None:
            left.append(f"cache/state/{stage}/{film_id}.json")
    msg = (f"{film_id}: catalog {len(updates)}행 영상 칸·match 비움"
           + (f", 문제 기록 지움({', '.join(cleared)})" if cleared else "")
           + (f", videos/ 지움: {', '.join(removed)}" if removed else ""))
    if kept:
        msg += (f" | videos/ 에 남김(다른 하드링크가 없어 원본이거나 복사본일 수 있음, 지우려면 --force): "
                f"{', '.join(kept)}")
    if left:
        msg += " | 지우지 않고 남긴 결과·캐시: " + ", ".join(left)
    return ("ok" if updates or cleared or removed else "skip"), msg


def run(args) -> int:
    vids = list(getattr(args, "paths", None) or [])
    docs = [d for d in (getattr(args, "docs", None) or "").split(",") if d]
    drop = bool(getattr(args, "unregister", False))
    if len(docs) != 1 or len(vids) != (0 if drop else 1) or (drop and getattr(args, "lang", None)):
        print(USAGE, file=sys.stderr, flush=True)
        return 1
    t0 = time.time()
    force = bool(getattr(args, "force", False))
    try:
        if drop:
            st, msg = unregister(docs[0], force=force)
        else:
            st, msg = register(vids[0], docs[0], getattr(args, "lang", None), force=force)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {docs[0]}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"{STAGE}: {{'error': 1}} ({time.time() - t0:.0f}초)", flush=True)
        return 1
    print(f"[1/1] {docs[0]} {st}: {msg}", flush=True)
    print(f"{STAGE}: {{'{st}': 1}} ({time.time() - t0:.0f}초)", flush=True)
    return 0
