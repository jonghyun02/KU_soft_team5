"""구글 번역 (논문 3.1절 sts 앞 한→영, 4.2절 영어 가사 영→한).

논문은 "Google Translator" 라고만 했다. 공개 웹 번역 주소(translate.googleapis.com, client=gtx)를 쓰고
결과를 sqlite 에 캐시해 같은 글을 두 번 보내지 않는다.
여러 줄은 줄바꿈으로 이어 한 번에 보내고, 돌아온 줄 수가 다르면 줄마다 따로 보낸다.

무료 주소는 많이 보내면 IP 가 몇 시간 막힌다(429, 2026-11-22 약 100만 자 뒤). 그래서 기본은
한 번에 하나, 요청 사이 2초, 막히면 포기하지 않고 최대 30분 간격으로 다시 해 보며 기다린다 (patient).
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time

import requests

from .config import CACHE

URL = "https://translate.googleapis.com/translate_a/single"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
MAX_CHARS = 4000


class GoogleTranslator:
    def __init__(self, db_path=None, min_interval: float = 2.0, workers: int = 1, patient: bool = True):
        db_path = db_path or (CACHE / "gtranslate.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.min_interval = min_interval
        self._last = 0.0
        self.n_requests = 0
        self.workers = workers
        self.patient = patient
        self.blocked_since = None

    @staticmethod
    def _key(text: str, sl: str, tl: str) -> str:
        return hashlib.sha1(f"{sl}|{tl}|{text}".encode("utf-8")).hexdigest()

    def _get(self, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM t WHERE k=?", (k,)).fetchone()
        return None if r is None else r[0]

    def _put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO t VALUES (?,?)", (k, v))
            self.db.commit()

    def _blocked(self, why: str, delay: float) -> float:
        if self.blocked_since is None:
            self.blocked_since = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] 구글 번역 막힘({why}), 기다렸다 다시 함", flush=True)
        time.sleep(delay + random.random() * 5)
        return min(delay * 2, 1800)

    def _request(self, text: str, sl: str, tl: str) -> str:
        delay = 10.0
        attempt = 0
        while True:
            attempt += 1
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.session.post(URL, params={"client": "gtx", "sl": sl, "tl": tl, "dt": "t"},
                                      data={"q": text}, timeout=30)
                self.n_requests += 1
                if r.status_code == 200:
                    if self.blocked_since is not None:
                        mins = (time.time() - self.blocked_since) / 60
                        print(f"[{time.strftime('%H:%M:%S')}] 구글 번역 다시 됨 ({mins:.0f}분 기다림)", flush=True)
                        self.blocked_since = None
                    data = json.loads(r.text)
                    return "".join(seg[0] for seg in (data[0] or []) if seg and seg[0])
                if r.status_code in (429, 503):
                    if not self.patient and attempt >= 12:
                        break
                    delay = self._blocked(str(r.status_code), delay)
                    continue
                r.raise_for_status()
            except (requests.RequestException, ValueError) as e:
                if not self.patient and attempt >= 12:
                    break
                delay = self._blocked(type(e).__name__, delay)
        raise RuntimeError(f"구글 번역 실패: {text[:60]!r}")

    def translate_one(self, text: str, sl: str, tl: str) -> str:
        if not text.strip():
            return text
        k = self._key(text, sl, tl)
        v = self._get(k)
        if v is None:
            v = self._request(text, sl, tl)
            self._put(k, v)
        return v

    def translate_lines(self, lines: list[str], sl: str, tl: str, progress: bool = False) -> list[str]:
        """줄 목록을 번역. 캐시에 없는 줄만 (중복 없이) 묶어서 보낸다."""
        out: list[str | None] = [None] * len(lines)
        todo: list[int] = []
        for i, line in enumerate(lines):
            if not line.strip():
                out[i] = line
                continue
            v = self._get(self._key(line, sl, tl))
            if v is None:
                todo.append(i)
            else:
                out[i] = v
        first_of: dict[str, int] = {}
        dup: list[int] = []
        uniq: list[int] = []
        for i in todo:
            if lines[i] in first_of:
                dup.append(i)
            else:
                first_of[lines[i]] = i
                uniq.append(i)
        todo = uniq
        bar = None
        if progress and todo:
            from tqdm import tqdm
            bar = tqdm(total=sum(len(lines[i]) + 1 for i in todo), desc=f"구글 번역 {sl}->{tl} (글자)", unit="ch")
        # 묶음 만들기 (한 요청 MAX_CHARS 자 이하)
        batches: list[list[int]] = []
        batch: list[int] = []
        size = 0
        for i in todo:
            n = len(lines[i]) + 1
            if batch and size + n > MAX_CHARS:
                batches.append(batch)
                batch, size = [], 0
            batch.append(i)
            size += n
        if batch:
            batches.append(batch)

        def run(b: list[int]) -> list[tuple[int, str]]:
            src = [lines[i] for i in b]
            res = self._request("\n".join(src), sl, tl).split("\n") if len(b) > 1 else None
            if res is not None and len(res) == len(b):
                return [(i, t.strip()) for i, t in zip(b, res)]
            return [(i, self._request(lines[i], sl, tl)) for i in b]

        # 요청 대기 시간이 대부분이라 몇 개를 동시에 보낸다. 캐시 쓰기는 이 스레드에서.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for res in ex.map(run, batches):
                for i, t in res:
                    out[i] = t
                    self._put(self._key(lines[i], sl, tl), t)
                if bar is not None:
                    bar.update(sum(len(lines[i]) + 1 for i, _ in res))
        for i in dup:
            out[i] = out[first_of[lines[i]]]
        if bar is not None:
            bar.close()
        return out  # type: ignore[return-value]
