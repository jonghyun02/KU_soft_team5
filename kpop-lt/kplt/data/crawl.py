"""저자 저장소 crawl.py 를 그대로 옮긴 가사 수집기.

저자 코드와 같은 것
- meta.csv 를 돌며 같은 검색어 순서로 Genius 를 찾는다 (english version, english ver, international ver, 제목만 ...).
- 비공식 영어 번역은 processed_lyrics 에 있으면 그것을 쓰고, 없으면 URL 의 유튜브 설명란이나 블로그 본문을 쓴다.
- Genius 검색 결과 고르기는 저자가 쓴 lyricsgenius.search_song 규칙을 옮겼다
  (제목이 같은 곡, 없으면 가사가 있는 첫 곡, 트랙리스트·크레딧 같은 글은 뺌).

저자 코드와 다른 것 (README 에 표시)
- lyricsgenius 대신 같은 공개 주소(genius.com/api/search/multi)와 곡 페이지를 직접 읽는다. 토큰 없이 된다.
- youtubesearchpython 은 깨져 있어 yt-dlp 로 설명란을 읽는다. 저자 utils.crawl_blog 는 공개되지 않아 새로 썼다.
- 실패한 곡은 멈추지 않고 failed.json 에 적는다 (저자 코드는 영어 쪽 실패에서 멈춤).

실행: python -m kplt.data.crawl [--shard 0 --n-shards 3]  (조각마다 다른 곡, 따로 띄워 병렬)
이미 받은 파일은 지금 정렬기로 다시 채점해 복원율 0.85 미만이면 다시 찾고, 더 나을 때만 덮어쓴다.
"""
from __future__ import annotations

import csv
import json
import re
import string
import time
import unicodedata

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from ..config import RAW, VENDOR
from ..text import load_meta
from .match import recovery_rate

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
OUT = RAW / "unprocessed_lyrics"
EXCLUDED_TERMS = ["track\\s?list", "album art(work)?", "liner notes", "booklet", "credits", "interview",
                  "skit", "instrumental", "setlist"]


def clean_str(s: str) -> str:
    """lyricsgenius.utils.clean_str 와 같음."""
    punct = string.punctuation + "’" + "​"
    s = s.translate(str.maketrans("", "", punct)).strip().lower()
    return unicodedata.normalize("NFKC", s)


class Genius:
    def __init__(self, sleep: float = 0.5):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.sleep = sleep

    def _get(self, url, **kw):
        delay = 5
        for _ in range(8):
            time.sleep(self.sleep)
            try:
                r = self.s.get(url, timeout=30, **kw)
            except requests.RequestException:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503):
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            r.raise_for_status()
        raise RuntimeError(f"Genius 요청 실패: {url}")

    def search_all(self, term: str) -> dict:
        r = self._get("https://genius.com/api/search/multi", params={"q": term, "per_page": 5})
        return r.json()["response"]

    @staticmethod
    def _is_lyrics(song: dict) -> bool:
        if song.get("lyrics_state") != "complete" or song.get("instrumental"):
            return False
        expr = "".join(f"({t})|" for t in EXCLUDED_TERMS).strip("|")
        return not re.search(expr, song["title"].lower())

    def _pick(self, response: dict, title: str) -> dict | None:
        """lyricsgenius _get_item_from_search_response(type_='song', result_type='title')."""
        sections = response["sections"]
        top_hits = sections[0]["hits"] if sections else []
        secs = sorted(sections, key=lambda s: s["type"] == "song", reverse=True)
        hits = [h for h in top_hits if h["type"] == "song"]
        hits.extend(h for s in secs for h in s["hits"] if h["type"] == "song")
        for h in hits:
            if clean_str(h["result"]["title"]) == clean_str(title):
                return h["result"]
        for h in hits:
            if self._is_lyrics(h["result"]):
                return h["result"]
        return hits[0]["result"] if hits else None

    def lyrics(self, url: str) -> str:
        html = self._get(url).text.replace("<br/>", "\n").replace("<br>", "\n")
        soup = BeautifulSoup(html, "lxml")
        divs = soup.select('div[data-lyrics-container="true"]')
        for d in divs:
            for ex in d.select('[data-exclude-from-selection="true"]'):
                ex.decompose()
        return "\n".join(d.get_text() for d in divs).strip("\n")

    def search_song(self, title: str, artist: str = "") -> dict | None:
        """lyricsgenius.Genius.search_song 처럼 (곡 정보 + lyrics) 를 돌려준다. 못 찾으면 None."""
        term = f"{title} {artist}".strip()
        song = self._pick(self.search_all(term), title)
        if song is None or not self._is_lyrics(song):
            return None
        text = self.lyrics(song["url"])
        if not text.strip():
            return None
        return {"query": term, "url": song["url"], "full_title": song.get("full_title"), "lyrics": text}


def youtube_description(url: str) -> str:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("description") or ""


def crawl_blog(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text.replace("<br/>", "\n").replace("<br>", "\n"), "lxml")
    body = soup.select_one("div.post-body") or soup.select_one("article")
    return body.get_text("\n") if body else ""


def _n(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", str(s)).lower())


def _artist_keys(artist: str) -> list[str]:
    """'Red Velvet (레드벨벳)' -> ['redvelvet레드벨벳', 'redvelvet', '레드벨벳']"""
    keys = {_n(artist)}
    keys.update(_n(p) for p in re.split(r"[()\[\]&,/]| x | feat\.? ", artist) if len(_n(p)) >= 2)
    return [k for k in keys if k]


def candidates(g: Genius, queries: list[tuple[str, str]], want_title: str, artist: str, lang: str):
    """[저자 코드와 다름] 모든 검색어의 결과를 모은 뒤 제목·가수가 맞는 순서로 내놓는다.

    저자 검색어(english version 등)를 그대로 쓰되, 검색 결과 맨 위만 믿지 않는다.
    점수: 제목 같음 3 / 포함 2, 가수 이름 들어 있음 2, 저자 규칙이 고른 곡 1,
    영어 쪽은 제목에 english·eng ver 가 있으면 1, 한국어 쪽은 english·translation·romaniz 가 있으면 -3,
    Genius 계정 글(Genius Korea, Genius English Translations ...)은 -5.
    """
    pool: dict[str, dict] = {}
    t_key = _n(want_title)
    a_keys = _artist_keys(artist)
    for title, art in queries:
        term = f"{title} {art}".strip()
        try:
            resp = g.search_all(term)
        except Exception as e:  # noqa: BLE001
            print("  검색 오류", term, e)
            continue
        pick = g._pick(resp, title)
        hits = [h["result"] for sec in resp["sections"] for h in sec["hits"] if h["type"] == "song"]
        for song in ([pick] if pick else []) + hits:
            if not g._is_lyrics(song):
                continue
            url = song["url"]
            ht = _n(song.get("title", ""))
            arts = _n(song.get("artist_names", "")) + _n(song.get("full_title", ""))
            low = (song.get("title", "") + " " + song.get("full_title", "")).lower()
            sc = 0
            if t_key and ht == t_key:
                sc += 3
            elif t_key and (t_key in ht or (len(ht) >= 3 and ht in t_key)):
                sc += 2
            if any(k in arts for k in a_keys):
                sc += 2
            if song is pick:
                sc += 1
            if lang == "en" and re.search(r"english|eng\.? ver", low):
                sc += 1
            if lang == "ko" and re.search(r"english|translation|romaniz", low):
                sc -= 3
            if (song.get("primary_artist") or {}).get("name", "").lower().startswith("genius"):
                sc -= 5
            prev = pool.get(url)
            if prev is None or sc > prev["score"]:
                pool[url] = {"query": term, "url": url, "full_title": song.get("full_title"),
                             "author_rule": song is pick, "score": sc}
    return sorted(pool.values(), key=lambda c: -c["score"])


def best_lyrics(g: Genius, queries, sym_sections, lang, want_title="", artist="", max_fetch=8, good=0.85):
    """[저자 코드와 다름] 후보 가사를 symbol.json 지문 복원율로 비교해 가장 잘 맞는 것을 고른다."""
    best = None
    for c in candidates(g, queries, want_title, artist, lang)[:max_fetch]:
        try:
            text = g.lyrics(c["url"])
        except Exception as e:  # noqa: BLE001
            print("  가사 오류", c["url"], e)
            continue
        if not text.strip():
            continue
        c = dict(c, lyrics=text, recovery=recovery_rate(text, sym_sections, lang))
        if best is None or c["recovery"] > best["recovery"]:
            best = c
        if c["recovery"] >= good:
            break
    return best


GOOD = 0.85


def _need(path, sym_sections, lang) -> float | None:
    """이미 받은 파일이 지금 정렬기로 GOOD 이상이면 None(건너뜀), 아니면 그 복원율(없으면 -1)."""
    if not path.exists():
        return -1.0
    r = recovery_rate(path.read_text(encoding="utf-8"), sym_sections, lang)
    return None if r >= GOOD else r


def main(shard: int = 0, n_shards: int = 1):
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = RAW / f"crawl_log_{shard}of{n_shards}.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    meta = [[m["LID"], m["ARTIST"], m["EN_TITLE"], m["KR_TITLE"], m["GENRE"], m["IS_OFFICIAL"], m["URL"]]
            for m in load_meta(VENDOR / "meta.csv").values()]
    meta = [m for i, m in enumerate(meta) if i % n_shards == shard]
    g = Genius()
    processed = VENDOR / "processed_lyrics"
    sym = json.load(open(VENDOR / "symbol.json", encoding="utf-8"))

    for item in tqdm(meta):
        lid, artist, en_title, kr_title, genre, is_official = item[0], item[1], item[2], item[3], item[4], item[-2]
        url = item[-1]
        rec = log.get(lid, {})
        en_path, kr_path = OUT / f"{lid}en.txt", OUT / f"{lid}kr.txt"
        unofficial = genre == "k-pop" and is_official[:1] != "1"

        # ---------- 영어 ----------
        old = None if (processed / f"{lid}en.txt").exists() else _need(en_path, sym[lid]["en"], "en")
        if old is not None and not (unofficial and en_path.exists()):
            en = None
            if unofficial:
                try:
                    if "youtu.be" in url or "youtube.com" in url:
                        en = {"source": url, "lyrics": youtube_description(url)}
                    elif "malloriemusic.blogspot.com" in url:
                        en = {"source": url, "lyrics": crawl_blog(url)}
                    if en:
                        en["recovery"] = recovery_rate(en["lyrics"], sym[lid]["en"], "en")
                except Exception as e:  # noqa: BLE001
                    print("  비공식 번역 오류", lid, url, e)
            elif genre == "k-pop":
                en = best_lyrics(g, [(en_title + " english version", artist), (en_title + " english ver", artist),
                                     (en_title + " international ver", artist), (en_title, artist)],
                                 sym[lid]["en"], "en", en_title, artist, good=GOOD)
            else:
                en = best_lyrics(g, [(en_title + " english", artist), (en_title, artist), (en_title, "")],
                                 sym[lid]["en"], "en", en_title, artist, good=GOOD)
            if en and en["lyrics"].strip() and en.get("recovery", 0) > old:
                en_path.write_text(en["lyrics"], encoding="utf-8")
                rec["en"] = {k: v for k, v in en.items() if k != "lyrics"}
            elif not en_path.exists():
                rec["en"] = None

        # ---------- 한국어 ----------
        old = _need(kr_path, sym[lid]["kr"], "ko")
        if old is not None:
            if genre == "k-pop":
                title = kr_title if len(kr_title) else en_title
                kr = best_lyrics(g, [(title, artist), (title + " korean ver", artist)], sym[lid]["kr"], "ko",
                                 title, artist, good=GOOD)
            else:
                qs = ([(kr_title, artist), (kr_title, "")] if len(kr_title) else [])
                qs += [(en_title + " korean", artist), (en_title + " korean version", "")]
                kr = best_lyrics(g, qs, sym[lid]["kr"], "ko", kr_title or en_title, artist, good=GOOD)
            if kr and kr.get("recovery", 0) > old:
                kr_path.write_text(kr["lyrics"], encoding="utf-8")
                rec["kr"] = {k: v for k, v in kr.items() if k != "lyrics"}
            elif not kr_path.exists():
                rec["kr"] = None
        log[lid] = rec
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    miss_en = [l for l in log if not (OUT / f"{l}en.txt").exists() and not (processed / f"{l}en.txt").exists()]
    miss_kr = [l for l in log if not (OUT / f"{l}kr.txt").exists()]
    (RAW / f"failed_{shard}of{n_shards}.json").write_text(json.dumps({"en": miss_en, "kr": miss_kr}, indent=1),
                                                          encoding="utf-8")
    print(f"영어 못 받음 {len(miss_en)}, 한국어 못 받음 {len(miss_kr)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    main(a.shard, a.n_shards)
