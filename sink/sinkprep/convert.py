"""convert 단계: hwp3/hwp5/hwpx/doc/docx/rtf -> cache/pdf/{doc_id}/{k}.pdf (한컴오피스 2024 자동화).

흐름(문서 하나, 파일 k 마다)
  1. 원본을 cache/convert_work/{doc_id}_{k}.{형식 확장자} 로 복사(원본 폴더는 읽기만).
     RTF 복사본은 prepare_rtf 로 손질(종이 크기 없는 RTF 의 \\sectd 제거), HWP5 는 문서 스크립트가
     기본 틀인지 검사해서 아니면 한컴으로 열지 않는다.
  2. 별도 프로세스 일꾼(sinkprep.hancom_worker)이 한글로 열어 PDF 로 저장.
     부모는 파일마다 시간 제한(TIMEOUT)을 두고, 넘으면 일꾼과 그 한글(Hwp.exe)을 죽이고
     다음 파일부터 새 일꾼을 띄운다. 일꾼 하나가 여러 파일을 차례로 처리한다.
  3. PDF 를 PyMuPDF 로 열어 쪽 수·글자 수를 확인(빈 PDF 는 실패로 침).
  4. 한컴이 실패하면 대체 경로로 글만 뽑아 cache/pdf/{doc_id}/{k}.txt:
       hwp3 -> sinkprep.hwp3text(HWP 3.0 문단 리스트 직접 읽기),
       hwp5 -> olefile BodyText 문단(안 되면 PrvText 미리보기), hwpx -> section XML,
       doc -> antiword, rtf -> striprtf, docx -> python-docx.
  5. cache/convert_log.jsonl 에 (doc_id, k)마다 최신 결과 한 줄.
     둘 다 실패하면 util.set_problems('convert', doc_id, [extract_broken]).

다시 돌리기: 지문(변환기 버전 + 파일별 원본 sha1·형식)이 같고 결과 파일이 그대로면 건너뛴다.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

from . import paths, util

STAGE = "convert"
VERSION = "0.2.0"
CONVERT_FORMATS = ("hwp3", "hwp5", "hwpx", "doc", "docx", "rtf")
EXT = {"hwp3": ".hwp", "hwp5": ".hwp", "hwpx": ".hwpx", "doc": ".doc", "docx": ".docx", "rtf": ".rtf"}
TIMEOUT = 120.0            # 파일 하나 한컴 변환 시간 제한(초)
START_TIMEOUT = 60.0       # 일꾼(한글) 시작 시간 제한
RECYCLE_EVERY = 40         # 이 수만큼 처리하면 한글을 새로 띄움(메모리·상태 누적 방지)
MIN_PDF_CHARS = 20         # PDF 전체 글자(공백 제외)가 이보다 적으면 빈 PDF 로 봄
ANTIWORD = Path(r"C:\Program Files\Git\mingw64\bin\antiword.exe")
WORK = paths.CACHE / "convert_work"
LOG = paths.CACHE / "convert_log.jsonl"


# ============================================================ 프로세스 정리

def _proc(pid: int):
    import psutil
    try:
        return psutil.Process(pid)
    except Exception:
        return None


def kill_tree(pid: int | None) -> None:
    """pid 와 그 자식들을 죽인다(가상환경 python.exe 는 실제 파이썬을 자식으로 띄우므로 트리째)."""
    if not pid:
        return
    import psutil
    p = _proc(pid)
    if p is None:
        return
    try:
        kids = p.children(recursive=True)
    except Exception:
        kids = []
    for q in kids + [p]:
        try:
            q.kill()
        except Exception:
            pass
    psutil.wait_procs(kids + [p], timeout=10)


def automation_hwp_pids() -> set[int]:
    """자동화로 뜬 한글(Hwp.exe -Automation) PID. 사용자가 직접 연 한글은 포함하지 않는다."""
    import psutil
    out = set()
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() == "hwp.exe" and "-Automation" in (p.info["cmdline"] or []):
                out.add(p.pid)
        except Exception:
            pass
    return out


def kill_hwp(pid: int | None) -> None:
    """한글과 그 한글이 띄운 한워드를 죽인다."""
    if not pid:
        return
    import psutil
    p = _proc(pid)
    try:
        if p is None or (p.name() or "").lower() != "hwp.exe":   # 이미 끝났거나 PID 가 재사용됨
            return
    except Exception:
        return
    for q in psutil.process_iter(["name", "ppid"]):
        if (q.info["name"] or "").lower() == "hword.exe" and q.info["ppid"] == pid:
            try:
                q.kill()
            except Exception:
                pass
    kill_tree(pid)


# ============================================================ 일꾼

class WorkerError(RuntimeError):
    pass


class HancomWorker:
    """hancom_worker 프로세스 하나와 그 한글 하나."""

    def __init__(self, start_timeout: float = START_TIMEOUT):
        self.proc: subprocess.Popen | None = None
        self.hwp_pid: int | None = None
        self.worker_pid: int | None = None
        self.version: str | None = None
        self.done = 0
        self._q: queue.Queue = queue.Queue()
        before = automation_hwp_pids()
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "sinkprep.hancom_worker"],
            cwd=str(paths.ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            msg = self._get(start_timeout)
            if msg.get("event") != "ready":
                raise WorkerError(f"start_failed:{msg.get('error')}")
        except WorkerError:
            # 시작 중 멈춤·실패: 그 사이 새로 뜬 자동화 한글을 이 일꾼 것으로 보고 정리
            self.kill()
            for pid in automation_hwp_pids() - before:
                kill_hwp(pid)
            raise
        self.hwp_pid = msg.get("hwp_pid")
        self.worker_pid = msg.get("pid")
        self.version = msg.get("version")
        self.module = msg.get("module")

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self._q.put(json.loads(line))
                except ValueError:
                    pass
        self._q.put({"event": "eof"})

    def _get(self, timeout: float) -> dict:
        try:
            msg = self._q.get(timeout=timeout)
        except queue.Empty:
            raise WorkerError("timeout") from None
        if msg.get("event") == "eof":
            raise WorkerError("worker_exited")
        return msg

    def convert(self, job: dict, timeout: float = TIMEOUT) -> dict:
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise WorkerError(f"worker_pipe:{e}") from None
        while True:
            msg = self._get(timeout)
            if msg.get("event") == "done" and msg.get("id") == job["id"]:
                self.done += 1
                return msg

    def close(self, grace: float = 15.0) -> None:
        """한글을 정상 종료시키고, 남으면 죽인다."""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            self.proc.wait(timeout=grace)
        except Exception:
            pass
        self._wait_hwp_exit(grace)
        self.kill()

    def _wait_hwp_exit(self, grace: float) -> None:
        if not self.hwp_pid:
            return
        p = _proc(self.hwp_pid)
        if p is None:
            return
        try:
            p.wait(timeout=grace)
        except Exception:
            pass

    def kill(self) -> None:
        kill_hwp(self.hwp_pid)
        if self.proc is not None:
            if self.proc.poll() is None:
                kill_tree(self.proc.pid)
            # 런처(가상환경 python.exe)가 먼저 끝나도 실제 파이썬(일꾼)이 남을 수 있다
            q = _proc(self.worker_pid) if self.worker_pid else None
            try:
                if q is not None and "sinkprep.hancom_worker" in " ".join(q.cmdline()):
                    kill_tree(q.pid)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None


class WorkerPool:
    """일꾼 하나를 필요할 때 띄우고, 멈추면 버리고 새로 띄운다."""

    def __init__(self, timeout: float = TIMEOUT, recycle_every: int = RECYCLE_EVERY):
        self.timeout = timeout
        self.recycle_every = recycle_every
        self.w: HancomWorker | None = None
        self.restarts = 0
        self.version: str | None = None
        self.start_error: str | None = None

    def convert(self, src: Path, fmt: str, out: Path, job_id: str) -> dict:
        if self.w is not None and self.w.done >= self.recycle_every:
            self.w.close()
            self.w = None
        if self.w is None:
            try:
                self.w = HancomWorker()
                self.version = self.w.version
            except WorkerError as e:
                self.start_error = str(e)
                return {"ok": False, "error": f"worker_start:{e}", "seconds": 0.0}
        job = {"id": job_id, "src": str(src), "fmt": fmt, "out": str(out)}
        try:
            return self.w.convert(job, self.timeout)
        except WorkerError as e:
            # 멈춤·죽음: 일꾼과 한글을 죽이고 다음 파일은 새 일꾼으로
            self.w.kill()
            self.w = None
            self.restarts += 1
            return {"ok": False, "error": f"hancom_{e}", "seconds": self.timeout}

    def close(self) -> None:
        if self.w is not None:
            self.w.close()
            self.w = None


# ============================================================ 대체 경로(글만)

def _hwp5_records(data: bytes):
    i, n = 0, len(data)
    while i + 4 <= n:
        h = int.from_bytes(data[i:i + 4], "little")
        i += 4
        tag, size = h & 0x3FF, (h >> 20) & 0xFFF
        if size == 0xFFF:
            if i + 4 > n:
                return
            size = int.from_bytes(data[i:i + 4], "little")
            i += 4
        yield tag, data[i:i + size]
        i += size


HWPTAG_PARA_TEXT = 0x10 + 51
_HWP_CHAR_CTRL = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}   # 1글자짜리 제어 문자


def _hwp5_para_text(b: bytes) -> str:
    out = []
    i, n = 0, len(b) // 2
    while i < n:
        c = int.from_bytes(b[2 * i:2 * i + 2], "little")
        if c >= 32:
            out.append(chr(c))
            i += 1
        elif c in _HWP_CHAR_CTRL:
            if c == 10:
                out.append("\n")
            elif c in (24, 30, 31):
                out.append(" " if c != 24 else "-")
            i += 1
        else:                       # 확장·인라인 제어: 8글자 차지
            if c == 9:
                out.append("\t")
            i += 8
    return "".join(out)


def hwp5_script_is_default(path: Path) -> bool:
    """HWP5 문서 스크립트가 한글 기본 틀(빈 OnDocument_New)뿐인지. 아니면 한컴으로 열지 않는다."""
    import olefile
    with olefile.OleFileIO(str(path)) as ole:
        hdr = ole.openstream("FileHeader").read()
        flags = int.from_bytes(hdr[36:40], "little")
        if flags & 0b1000:          # 스크립트 저장 표시
            return False
        if not ole.exists("Scripts/DefaultJScript"):
            return True
        raw = ole.openstream("Scripts/DefaultJScript").read()
        try:
            data = zlib.decompressobj(-15).decompress(raw) if flags & 1 else raw
        except zlib.error:
            data = raw
    text = data.decode("utf-16-le", "ignore")
    body = re.sub(r"var Documents = XHwpDocuments;|var Document = Documents\.Active_XHwpDocument;"
                  r"|function OnDocument_New\(\)\s*\{\s*//todo : \s*\}", "", text)
    return not re.search(r"[A-Za-z]{3,}", body)


def hwp5_text(path: Path) -> tuple[str, str]:
    """(글, 방법). BodyText 문단 글, 안 되면 PrvText(앞부분 미리보기)."""
    import olefile
    with olefile.OleFileIO(str(path)) as ole:
        flags = int.from_bytes(ole.openstream("FileHeader").read()[36:40], "little")
        if flags & 0b10:
            raise ValueError("hwp5_password")
        secs = sorted((e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"
                       and e[1].startswith("Section")), key=lambda e: int(e[1][7:] or 0))
        paras: list[str] = []
        for e in secs:
            raw = ole.openstream(e).read()
            try:   # 끝이 잘린 구역도 읽을 수 있는 만큼 읽는다
                data = zlib.decompressobj(-15).decompress(raw) if flags & 1 else raw
            except zlib.error:
                continue
            for tag, body in _hwp5_records(data):
                if tag == HWPTAG_PARA_TEXT:
                    paras.append(_hwp5_para_text(body))
        text = "\n".join(paras)
        if text.strip():
            return text, "fallback_hwp5_bodytext"
        if ole.exists("PrvText"):
            prv = ole.openstream("PrvText").read().decode("utf-16-le", "ignore")
            if prv.strip():
                return prv, "fallback_hwp5_prvtext"
    raise ValueError("hwp5_no_text")


def hwpx_text(path: Path) -> tuple[str, str]:
    import zipfile
    from xml.etree import ElementTree as ET
    paras: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n, re.I)),
                       key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
        for n in names:
            root = ET.fromstring(z.read(n))
            for p in root.iter():
                if p.tag.endswith("}p"):
                    t = "".join(x.text or "" for x in p.iter() if x.tag.endswith("}t"))
                    paras.append(t)
    text = "\n".join(paras)
    if not text.strip():
        raise ValueError("hwpx_no_text")
    return text, "fallback_hwpx"


def doc_text(path: Path) -> tuple[str, str]:
    if not ANTIWORD.exists():
        raise ValueError("antiword_missing")
    path = Path(path).resolve()
    WORK.mkdir(parents=True, exist_ok=True)       # 원본 폴더(신뢰할 수 없는 내려받은 파일)를 현재 폴더로 두지 않음
    r = subprocess.run([str(ANTIWORD), "-m", "UTF-8.txt", "-w", "0", str(path)],
                       capture_output=True, timeout=120, cwd=str(WORK),
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = r.stdout.decode("utf-8", "replace")
    if r.returncode != 0 or not text.strip():
        raise ValueError(f"antiword_rc{r.returncode}:{r.stderr.decode('utf-8', 'replace')[:200]}")
    return text, "fallback_antiword"


def rtf_text(path: Path) -> tuple[str, str]:
    from striprtf.striprtf import rtf_to_text
    raw = path.read_bytes()
    m = re.search(rb"\\ansicpg(\d+)", raw[:4096])
    enc = f"cp{int(m.group(1))}" if m else "cp1252"
    try:
        "".encode(enc)
    except LookupError:
        enc = "cp1252"
    text = rtf_to_text(raw.decode("latin-1"), encoding=enc, errors="replace")
    if not text.strip():
        raise ValueError("rtf_no_text")
    return text, "fallback_rtf"


def docx_text(path: Path) -> tuple[str, str]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(str(path))
    out: list[str] = []

    def walk(parent):
        for block in parent.iter_inner_content():
            if isinstance(block, Paragraph):
                out.append(block.text)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = []
                    for cell in row.cells:
                        cells.append(" ".join(p.text for p in cell.paragraphs).strip())
                    out.append("\t".join(cells))
    walk(d)
    text = "\n".join(out)
    if not text.strip():
        raise ValueError("docx_no_text")
    return text, "fallback_docx"


def hwp3_text(path: Path) -> tuple[str, str] | tuple[str, str, str]:
    """HWP3 문단 글(sinkprep.hwp3text). 구조가 망가져 도중에 멈췄으면 셋째 값으로 그 까닭."""
    from . import hwp3text
    paras, stop = hwp3text.extract(path)
    text = "\n".join(paras)
    if not text.strip():
        raise ValueError(f"hwp3_no_text:{stop}" if stop else "hwp3_no_text")
    if stop:
        return text, "fallback_hwp3_text", f"hwp3_partial:{stop}"
    return text, "fallback_hwp3_text"


FALLBACK = {"hwp3": hwp3_text, "hwp5": hwp5_text, "hwpx": hwpx_text, "doc": doc_text, "rtf": rtf_text,
            "docx": docx_text}


def fallback_text(path: Path, fmt: str, notes: list[str] | None = None) -> tuple[str, str]:
    """(글, 방법). 대체 함수가 덧붙인 설명(예: 일부만 읽음)은 notes 에 담는다."""
    fn = FALLBACK.get(fmt)
    if fn is None:
        raise ValueError(f"no_fallback:{fmt}")
    text, method, *extra = fn(path)
    if notes is not None:
        notes.extend(extra)
    text = util.nfc(text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", ""))
    return text, method


# ============================================================ 복사본 손질

_SECTD_RE = re.compile(rb"\\sectd(?![a-z])[ ]?")


def prepare_rtf(path: Path) -> str | None:
    """작업 복사본 RTF 손질(원본은 그대로). 손질했으면 설명을 돌려준다.

    종이 크기(\\paperw)가 없는 RTF 에서 문단 안 \\sectd(구역 초기화)가 나오면 한글 2024 의 RTF
    필터가 쪽을 수만 개(가을로 [55354]: 28985쪽)로 나누고 PDF 저장에서 멈춘다(실험으로 확인).
    이 경우 \\sectd 만 지우면 앞에 적힌 여백이 살아서 정상 쪽 수(61쪽)로 변환된다.
    """
    data = path.read_bytes()
    if b"\\paperw" in data or not _SECTD_RE.search(data):
        return None
    paths.assert_not_raw(path)
    path.write_bytes(_SECTD_RE.sub(b"", data))
    return "rtf_sectd_removed(no_paperw)"


# ============================================================ PDF 확인

def pdf_stats(path: Path) -> tuple[int, int]:
    """(쪽 수, 공백 뺀 글자 수)."""
    import pymupdf
    with pymupdf.open(str(path)) as d:
        chars = 0
        for page in d:
            chars += sum(1 for ch in page.get_text("text") if not ch.isspace())
        return d.page_count, chars


# ============================================================ 문서 목록

_POST_RE = re.compile(r"\[(\d+)\]")


def docs_from_paths(files: list[str]) -> list[dict]:
    """catalog 없이 파일 경로로 돌릴 때: 파일 이름의 [글번호]로 doc_id 를 만든다."""
    from . import detect
    by_id: dict[str, dict] = {}
    for f in files:
        p = Path(f).resolve()
        m = _POST_RE.search(p.name)
        doc_id = f"fm{m.group(1)}" if m else "file_" + util.sha1_file(p)[:10]
        try:
            rel = paths.rel(p)
        except ValueError:
            rel = str(p)
        d = by_id.setdefault(doc_id, {"doc_id": doc_id, "film_id": "", "status": "pending",
                                      "files_list": [], "file_formats_list": []})
        d["files_list"].append(rel)
        d["file_formats_list"].append(detect.real_format(p))
    return list(by_id.values())


def load_docs(args) -> list[dict]:
    files = list(getattr(args, "paths", None) or [])
    if files:
        docs = docs_from_paths(files)
    else:
        from . import catalog
        docs = catalog.load()
    want = getattr(args, "docs", None)
    if want:
        keep = {x.strip() for x in want.split(",") if x.strip()}
        docs = [d for d in docs if d["doc_id"] in keep]
    docs = [d for d in docs if d.get("status") != "excluded"
            and any(f in CONVERT_FORMATS for f in (d.get("file_formats_list") or []))]
    limit = getattr(args, "limit", None)
    if limit:
        docs = docs[:limit]
    return docs


# ============================================================ 기록

def read_log() -> dict[tuple[str, int], dict]:
    if not LOG.exists():
        return {}
    out = {}
    for r in util.read_jsonl(LOG):
        out[(r["doc_id"], int(r["k"]))] = r
    return out


def write_log(log: dict[tuple[str, int], dict]) -> None:
    util.write_jsonl(LOG, [log[k] for k in sorted(log)])


def _place(src: Path, dst: Path) -> None:
    """작업 폴더의 결과를 제자리로(같은 드라이브 os.replace = 원자적)."""
    paths.assert_not_raw(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


# ============================================================ 교차 검사

# 대체 경로 글의 글자(문자·숫자) 중 한컴 PDF 에 없는 비율(loss).
# 문서 전체에서 정상 변환은 0.002 아래였고 PDF 가 글을 잃은 문서는 0.015~0.16 이었다.
PDF_LOSS_WARN = 0.01     # 넘으면 needs_review 에 올림(PDF 는 그대로 씀: 레이아웃이 있어야 구조 파싱이 됨)
PDF_LOSS_MAX = 0.03      # 넘으면 PDF 대신 대체 경로 글을 씀(내용이 더 중요)
LOSS_MIN_CHARS = 500     # 대체 글이 이보다 짧으면(대체 경로가 거의 못 읽음) 검사하지 않는다


def letters_loss(alt: str, pdf_text: str) -> float:
    """대체 글에만 있고 PDF 글에는 없는 글자 비율(문자·숫자 다중집합 차 / 대체 글 글자 수).

    구분선(─)·기호는 한컴 글자층이 빈칸 글리프로 잃기도 해서 문자·숫자만 센다."""
    from collections import Counter
    a = Counter(c for c in util.nfc(alt) if c.isalnum())
    b = Counter(c for c in util.nfc(pdf_text) if c.isalnum())
    return sum((a - b).values()) / max(1, sum(a.values()))


def crosscheck(row: dict, src: Path) -> dict:
    """한컴 PDF 를 다른 방법(fallback_text)으로 읽은 원본 글과 견준다. 한컴 2024 가 일부 HWP5·doc 에서
    줄을 쪽 밖으로 밀어내거나 빠뜨린 PDF 를 만든다(예: 사랑을 놓치다 fm55169, Fargo fm56063).
    PDF 가 글을 잃었으면 PDF 를 {k}.incomplete.pdf 로 옮기고 대체 글을 {k}.txt 로 쓴다(레이아웃은 잃음).
    row 에 loss 를 남긴다(검사 못 하면 None)."""
    import pymupdf
    doc_id, k, fmt = row["doc_id"], row["k"], row["real_format"]
    pdf_out = paths.PDF_CACHE / doc_id / f"{k}.pdf"
    row["loss"] = None
    if row.get("method") != "hancom" or not pdf_out.exists() or fmt not in FALLBACK:
        return row
    try:
        alt, method = fallback_text(src, fmt)[:2]
    except Exception:  # noqa: BLE001 - 대체 경로가 못 읽으면 견줄 수 없음
        return row
    if sum(1 for c in alt if c.isalnum()) < LOSS_MIN_CHARS:
        return row
    with pymupdf.open(pdf_out) as d:
        loss = letters_loss(alt, "".join(pg.get_text() for pg in d))
    row["loss"] = round(loss, 4)
    if loss > PDF_LOSS_MAX:
        _place(pdf_out, paths.PDF_CACHE / doc_id / f"{k}.incomplete.pdf")
        util.write_text(paths.PDF_CACHE / doc_id / f"{k}.txt", alt)
        row["note"] = "; ".join(x for x in (row.get("note"), f"hancom_pdf_incomplete:loss={loss:.3f}") if x)
        row.update(method=method, pages=None)
    return row


def file_problems(doc: dict, row: dict) -> list[dict]:
    """변환 로그 행 하나 -> 문제 기록(needs_review 로 감)."""
    base = {"level": "doc", "doc_id": doc["doc_id"], "film_id": doc.get("film_id") or "",
            "item_id": "", "path": row["src"]}
    err = row.get("error") or ""
    if not row["ok"]:
        return [{**base, "problem": "extract_broken", "score": 0.0,
                 "detail": f"변환 실패(한컴·대체 경로 모두): {err}"}]
    if row["method"] != "hancom" and "hwp3_partial" in err:
        m = re.search(r"read=([\d.]+)", err)
        return [{**base, "problem": "extract_broken", "score": float(m.group(1)) if m else 0.5,
                 "detail": "한컴이 열지 못하고 HWP3 직접 해석도 파일이 망가진 곳에서 멈춤"
                           f"(앞부분만 읽음, 레이아웃 없음): {err}"}]
    if row["method"] == "hancom" and (row.get("loss") or 0) > PDF_LOSS_WARN:
        return [{**base, "problem": "other", "score": round(1 - row["loss"], 3),
                 "detail": f"한컴 PDF 가 원본 글 일부를 잃었을 수 있음(대체 경로 글의 글자 중 {100 * row['loss']:.1f}% 없음), "
                           "레이아웃 때문에 PDF 를 씀"}]
    if row["method"] != "hancom" and "hancom_pdf_incomplete" in (row.get("note") or ""):
        loss = row.get("loss") or 0
        return [{**base, "problem": "other", "score": round(1 - loss, 3),
                 "detail": f"한컴 PDF 가 글 일부를 잃음(원본 글자 중 {100 * loss:.1f}% 없음), "
                           f"대체 경로 글을 씀(레이아웃 없음, {row['method']})"}]
    if row["method"] != "hancom":
        return [{**base, "problem": "other", "score": 0.5,
                 "detail": f"한컴 변환 실패, 대체 경로 글만 있음(레이아웃 없음, {row['method']}): {err}"}]
    return []


# ============================================================ 변환

def convert_file(pool: WorkerPool | None, doc_id: str, k: int, relpath: str, fmt: str) -> dict:
    """파일 하나. 결과 로그 행을 돌려준다."""
    t0 = time.time()
    src_abs = paths.absp(relpath) if not Path(relpath).is_absolute() else Path(relpath)
    out_dir = paths.PDF_CACHE / doc_id
    pdf_out, txt_out = out_dir / f"{k}.pdf", out_dir / f"{k}.txt"
    WORK.mkdir(parents=True, exist_ok=True)
    stem = f"{doc_id}_{k}"
    work_src = WORK / f"{stem}{EXT[fmt]}"
    work_pdf = WORK / f"{stem}.pdf"
    paths.assert_not_raw(work_src)
    shutil.copyfile(src_abs, work_src)
    row = {"doc_id": doc_id, "k": k, "src": relpath, "real_format": fmt, "ok": False,
           "method": None, "pages": None, "error": None, "seconds": None, "note": None, "loss": None}
    errors: list[str] = []
    try:
        use_hancom = pool is not None
        if fmt == "rtf":
            row["note"] = prepare_rtf(work_src)
        if use_hancom and fmt == "hwp5":
            try:
                if not hwp5_script_is_default(work_src):
                    use_hancom = False
                    errors.append("hwp5_custom_script:hancom_skipped")
            except Exception as e:
                errors.append(f"hwp5_script_check:{e}")
        if use_hancom:
            _unlink(work_pdf)
            res = pool.convert(work_src, fmt, work_pdf, stem)
            if res.get("ok") and work_pdf.exists():
                try:
                    pages, chars = pdf_stats(work_pdf)
                except Exception as e:
                    pages, chars = 0, 0
                    errors.append(f"pdf_open:{e}")
                if pages > 0 and chars >= MIN_PDF_CHARS:
                    _place(work_pdf, pdf_out)
                    _unlink(txt_out)
                    _unlink(out_dir / f"{k}.incomplete.pdf")
                    row.update(ok=True, method="hancom", pages=pages)
                    if res.get("how") == "open_yes":       # '예'로 다시 열어서 된 문서
                        row["note"] = "; ".join(x for x in (row["note"], "hancom_open_yes") if x)
                    crosscheck(row, src_abs)
                else:
                    errors.append(f"hancom_empty_pdf:pages={pages},chars={chars}")
            else:
                errors.append(str(res.get("error") or "hancom_failed"))
            if res.get("windows"):
                errors.append("windows_seen:" + "|".join(res["windows"]))
        if not row["ok"] and pdf_out.exists() and any(t in e for e in errors
                                                      for t in ("hancom_timeout", "hancom_worker", "worker_start")):
            # 예전에 만든 PDF 가 있는데 이번에 한컴이 시간 초과·일꾼 오류·기동 실패: 일시적일 수 있으니 PDF 를 그대로 둔다
            row.update(ok=True, method="hancom", note="; ".join(x for x in (row["note"], "kept_previous_pdf") if x))
        if not row["ok"]:
            try:
                notes: list[str] = []
                text, method = fallback_text(work_src, fmt, notes)
                util.write_text(txt_out, text)
                _unlink(pdf_out)
                row.update(ok=True, method=method, pages=None)
                errors.extend(notes)
            except Exception as e:
                errors.append(f"fallback:{e}")
                _unlink(pdf_out)
                _unlink(txt_out)
    finally:
        _unlink(work_src)
        _unlink(work_pdf)
    row["error"] = "; ".join(errors) or None
    row["seconds"] = round(time.time() - t0, 3)
    return row


def doc_fp(doc: dict) -> dict:
    files = []
    for k, (rel, fmt) in enumerate(zip(doc["files_list"], doc["file_formats_list"])):
        if fmt in CONVERT_FORMATS:
            p = paths.absp(rel) if not Path(rel).is_absolute() else Path(rel)
            files.append([k, rel, fmt, util.sha1_file(p)])
    return {"version": VERSION, "files": files}


def outputs_present(doc_id: str, fp: dict, log: dict) -> bool:
    for k, _, _, _ in fp["files"]:
        r = log.get((doc_id, k))
        if r is None:
            return False
        if r.get("method") == "hancom" and not (paths.PDF_CACHE / doc_id / f"{k}.pdf").exists():
            return False
        if (r.get("method") or "").startswith("fallback") and not (paths.PDF_CACHE / doc_id / f"{k}.txt").exists():
            return False
    return True


def convert_doc(doc: dict, pool: WorkerPool | None, log: dict, force: bool = False) -> str:
    """문서 하나. 반환: skip | ok | partial | failed."""
    doc_id = doc["doc_id"]
    fp = doc_fp(doc)
    if not force and util.stage_fp(STAGE, doc_id) == fp and outputs_present(doc_id, fp, log):
        # 교차 검사를 넣기 전에 변환한 문서: 한컴을 다시 돌리지 않고 검사만 한 번 한다
        todo = [(k, rel) for k, rel, _, _ in fp["files"] if "loss" not in log[(doc_id, k)]]
        if not todo:
            return "skip"
        for k, rel in todo:
            crosscheck(log[(doc_id, k)], paths.absp(rel) if not Path(rel).is_absolute() else Path(rel))
        util.set_problems(STAGE, doc_id, [p for k, *_ in fp["files"] for p in file_problems(doc, log[(doc_id, k)])])
        write_log(log)
        return "checked"
    problems = []
    rows = []
    for k, rel, fmt, _ in fp["files"]:
        row = convert_file(pool, doc_id, k, rel, fmt)
        rows.append(row)
        log[(doc_id, k)] = row
        problems.extend(file_problems(doc, row))
    util.set_problems(STAGE, doc_id, problems)
    # 한글을 아예 못 띄운 경우(환경 문제)는 지문을 남기지 않아 다음 실행에서 다시 시도한다.
    transient = ("worker_start", "hancom_timeout", "hancom_worker")
    if not any(t in (r["error"] or "") for r in rows for t in transient):
        util.set_stage_fp(STAGE, doc_id, fp)
    write_log(log)
    if all(r["ok"] and r["method"] == "hancom" for r in rows):
        return "ok"
    return "failed" if not any(r["ok"] for r in rows) else "partial"


def run(args) -> int:
    docs = load_docs(args)
    force = bool(getattr(args, "force", False))
    workers = getattr(args, "workers", None)
    if workers and workers > 1:
        print("convert: 한글 자동화는 일꾼 1개로 차례로 처리합니다(--workers 무시).", flush=True)
    log = read_log()
    counts = {"skip": 0, "checked": 0, "ok": 0, "partial": 0, "failed": 0}
    pool = WorkerPool()
    t0 = time.time()
    try:
        for i, doc in enumerate(docs, 1):
            st = convert_doc(doc, pool, log, force=force)
            counts[st] += 1
            notes = [str(log.get((doc["doc_id"], k), {}).get("note")) for k in range(len(doc.get("files_list") or []))]
            if st not in ("skip", "checked") or any("hancom_pdf_incomplete" in n for n in notes):
                print(f"[{i}/{len(docs)}] {doc['doc_id']} {st} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        pool.close()
        _cleanup_work()
    print(f"convert: 문서 {len(docs)} - " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f", 일꾼 재시작 {pool.restarts}, 한글 {pool.version}, {time.time() - t0:.0f}초", flush=True)
    return 0


def _cleanup_work() -> None:
    if WORK.exists():
        for p in WORK.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
