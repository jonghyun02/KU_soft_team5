"""Genius 전체 덤프(HF sebastiandizon/genius-song-lyrics, 약 500만 곡, 언어 표시 포함)에서
논문이 출처를 밝히지 않은 세 묶음을 뽑는다. [해석] 출처 선택은 논문에 없음.

- nonsingable: 4.2절 "10,000 English lyrics randomly from the internet" -> language=en 무작위 1만 곡
- nsp: 4.3절 "English lyrics from 7,103 songs that are not included in our training and evaluation data"
       -> language=en, nonsingable 과 겹치지 않는 무작위 7,103 곡
- eval: 4.4절 "lyrics from 2,038 K-pop songs that are divided by sections" -> language=ko, tag=pop,
        [Verse] 같은 마디 머리줄로 두 마디 이상 나뉜 곡 중 무작위 2,038 곡

논문 데이터셋 1,000곡과 겹치는 곡은 모든 묶음에서 뺀다 (4.3절, 4.4절).
- 같은 곡 판정: 가수 이름(괄호 안 한글 이름 등으로 나눈 여러 표기) 중 하나가 덤프 가수에 들어 있고,
  덤프 제목이 데이터셋 영어·한국어 제목(5글자 이상)과 같거나 그것을 품으면 같은 곡 (Live, Remix, Ver. 도 걸림).
- 디즈니·뮤지컬 줄은 meta 의 가수 칸이 영화 이름이라 제목만으로 판정한다 (커버곡 제외).
[해석] 고른 기준
- 마디·줄은 Genius 머리줄과 빈 줄로 나눈다. 마디가 2개 이상이고 40줄이 넘는 마디가 없는 곡만 쓴다.
- Genius 계정 글(Genius Translations 등), 제목에 Translation·번역·Romaniz 가 있는 곡,
  가사 안에 Romanization·Hangul·English Translation 같은 표시 줄이 있는 곡은 뺀다.

실행: python -m kplt.data.dump
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys

import pandas as pd

from ..config import DATA, N_EVAL_SONGS, N_NONSINGABLE, N_NSP_SONGS, RAW, SEED, VENDOR
from ..text import SECTION_HEADER, has_hangul, load_meta, normalize_text, split_sections

DUMP = RAW / "genius_dump" / "song_lyrics.csv"


MAX_SECTION_LINES = 40
MARKER = re.compile(r"^\s*\[?\s*(romani[sz](ation|ed)( lyrics)?|hangul|english translation|translation|korean|english)\s*\]?\s*:?\s*$",
                    re.I | re.M)
BAD_TITLE = re.compile(r"translation|번역|romaniz", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", normalize_text(str(s)).lower())


def _artist_variants(artist: str) -> set[str]:
    out = {_norm(artist)}
    out.update(_norm(p) for p in re.split(r"[()\[\]&,/]| x | feat\.? ", artist, flags=re.I))
    return {a for a in out if len(a) >= 2}


_VER = re.compile(r"\b(live|remix(ed)?|ver\.?|version|edit|inst\.?|instrumental|acoustic|english|korean|japanese|"
                  r"nightmare|funk boost|feat\.?|ft\.?)\b", re.I)


def title_variants(title: str) -> set[str]:
    """제목의 여러 표기: 전체, 괄호 밖, 괄호 안, 라틴 글자만, 한글만, 그리고 각각에서 Live·Remix·Ver. 같은 말을 뺀 것."""
    t = normalize_text(str(title))
    out = {t, re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)}
    out.update(re.findall(r"\(([^)]*)\)", t))
    more = set()
    for x in out:
        more.add(re.sub(r"[가-힣]+", " ", x))
        more.add(re.sub(r"[^가-힣\s]+", " ", x))
    out |= more
    out |= {_VER.sub(" ", x) for x in out}
    return {n for n in (_norm(x) for x in out) if len(n) >= 2}


class DatasetSongs:
    """논문 데이터셋 곡과 같은 곡인지 판정.

    덤프 제목의 표기 중 하나가 데이터셋 제목 표기 중 하나와 같거나, 5글자 이상 데이터셋 표기를 품으면 제목이 같다고 본다.
    K-pop 은 가수 표기 하나가 덤프 가수에 들어 있어야 하고, 디즈니·뮤지컬은 제목만 본다.
    """

    def __init__(self):
        self.by_artist: list[tuple[set[str], set[str]]] = []
        self.title_only: set[str] = set()
        for m in load_meta(VENDOR / "meta.csv").values():
            titles = set()
            for t in (m["EN_TITLE"], m["KR_TITLE"]):
                if t:
                    titles |= title_variants(t)
            if m["GENRE"] == "k-pop":
                self.by_artist.append((_artist_variants(m["ARTIST"]), titles))
            else:
                self.title_only.update(titles)
        all_t = set(self.title_only).union(*(ts for _, ts in self.by_artist))
        self._exact = all_t
        longs = sorted((x for x in all_t if len(x) >= 5), key=len, reverse=True)
        self._re = re.compile("|".join(re.escape(x) for x in longs))

    @staticmethod
    def _title_hit(dump_vars: set[str], titles: set[str]) -> bool:
        return any(d == t or (len(t) >= 5 and t in d) for d in dump_vars for t in titles)

    def __call__(self, artist: str, title: str) -> bool:
        full = _norm(title)
        if not (self._re.search(full) or full in self._exact):
            dvars = title_variants(title)
            if not any(d in self._exact or self._re.search(d) for d in dvars):
                return False
        else:
            dvars = title_variants(title) | {full}
        a = _norm(artist)
        if self._title_hit(dvars, self.title_only):
            return True
        for variants, titles in self.by_artist:
            if any(v in a for v in variants) and self._title_hit(dvars, titles):
                return True
        return False


def has_headers(lyrics: str) -> bool:
    return sum(bool(SECTION_HEADER.match(l)) for l in lyrics.splitlines()) >= 2


def bad_page(artist: str, title: str, lyrics: str) -> bool:
    return (artist.lower().startswith("genius") or bool(BAD_TITLE.search(title))
            or bool(MARKER.search(lyrics)))


def good_structure(sections: list[list[str]]) -> bool:
    return len(sections) >= 2 and max(len(s) for s in sections) <= MAX_SECTION_LINES


def scan():
    """덤프를 조각으로 읽어 영어 후보 id 와 한국어 K-pop 후보를 모은다."""
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    same = DatasetSongs()
    en_ids, ko_rows = [], []
    cols = ["title", "tag", "artist", "lyrics", "id", "language"]
    for chunk in pd.read_csv(DUMP, usecols=cols, chunksize=200_000, dtype=str, keep_default_na=False,
                             engine="c", on_bad_lines="skip"):
        en = chunk[(chunk.language == "en")]
        en = en[~en.apply(lambda r: same(r.artist, r.title) or bad_page(r.artist, r.title, ""), axis=1)]
        en_ids.extend(en.id.tolist())
        ko = chunk[(chunk.language == "ko") & (chunk.tag == "pop")]
        for r in ko.itertuples():
            if (has_headers(r.lyrics) and has_hangul(r.lyrics) and not same(r.artist, r.title)
                    and not bad_page(r.artist, r.title, r.lyrics)
                    and good_structure(split_sections(r.lyrics))):
                ko_rows.append({"id": r.id, "artist": r.artist, "title": r.title, "lyrics": r.lyrics})
        print(f"  읽음: 영어 후보 {len(en_ids)}, 한국어 후보 {len(ko_rows)}", flush=True)
    return en_ids, ko_rows


def fetch_by_id(ids: set[str]) -> dict[str, dict]:
    out = {}
    for chunk in pd.read_csv(DUMP, usecols=["title", "artist", "lyrics", "id"], chunksize=200_000, dtype=str,
                             keep_default_na=False, engine="c", on_bad_lines="skip"):
        sub = chunk[chunk.id.isin(ids)]
        for r in sub.itertuples():
            out[r.id] = {"id": r.id, "artist": r.artist, "title": r.title, "lyrics": r.lyrics}
    return out


def to_song(row: dict) -> dict:
    return {"id": row["id"], "artist": row["artist"], "title": row["title"],
            "sections": split_sections(row["lyrics"])}


def usable_en(song: dict, lyrics: str) -> bool:
    return (good_structure(song["sections"]) and not has_hangul(" ".join(sum(song["sections"], [])))
            and not MARKER.search(lyrics))


def main():
    rng = random.Random(SEED)
    en_ids, ko_rows = scan()
    rng.shuffle(en_ids)
    # 쓸 수 없는 곡을 건너뛰도록 넉넉히 뽑아 둔다
    want = N_NONSINGABLE + N_NSP_SONGS
    pool_ids = en_ids[: int(want * 1.8)]
    rows = fetch_by_id(set(pool_ids))
    songs = [(to_song(rows[i]), rows[i]["lyrics"]) for i in pool_ids if i in rows]
    songs = [s for s, lyr in songs if usable_en(s, lyr)]
    nonsing, nsp = songs[:N_NONSINGABLE], songs[N_NONSINGABLE:N_NONSINGABLE + N_NSP_SONGS]
    assert len(nonsing) == N_NONSINGABLE and len(nsp) == N_NSP_SONGS, (len(nonsing), len(nsp))

    rng.shuffle(ko_rows)
    eval_songs = [to_song(r) for r in ko_rows[:N_EVAL_SONGS]]
    assert len(eval_songs) == N_EVAL_SONGS, len(eval_songs)

    DATA.mkdir(parents=True, exist_ok=True)
    for name, obj in [("nonsingable_en.json", nonsing), ("nsp_en.json", nsp), ("eval_kpop.json", eval_songs)]:
        (DATA / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        print(name, len(obj), "곡", sum(len(sec) for s in obj for sec in s["sections"]), "줄")


if __name__ == "__main__":
    main()
