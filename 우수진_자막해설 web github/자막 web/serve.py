"""
CBoE 검수 화면 로컬 서버 — 영상 탐색(Range 요청)을 지원하는 정적 서버.

사용법
  python serve.py                         # http://localhost:8000
  python serve.py --port 8080
  python serve.py --work work/sample --video media/movie.mp4

기본 http.server 는 Range 요청을 지원하지 않아 Chrome 에서 영상 탐색(건너뛰기)이
안 될 수 있어요. 이 서버는 그 부분을 해결합니다.
"""
import argparse
import http.server
import os
import re
import socketserver
import threading
import urllib.parse
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
        ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        ".vtt": "text/vtt; charset=utf-8", ".srt": "text/plain; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".md": "text/markdown; charset=utf-8",
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        if not m:
            return super().send_head()
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        try:
            while left > 0:
                chunk = source.read(min(64 * 1024, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 사용자가 탐색하면 브라우저가 연결을 끊는 게 정상
        finally:
            self._range_left = None

    def log_message(self, fmt, *args):
        if os.environ.get("CBOE_QUIET"):
            return
        super().log_message(fmt, *args)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def player_url(port, video=None, subs=None, work=None):
    q = {k: v for k, v in (("work", work), ("video", video), ("subs", subs)) if v}
    page = "studio.html" if q else "index.html"
    return f"http://localhost:{port}/{page}" + ("?" + urllib.parse.urlencode(q) if q else "")


def start(port=8000, background=False):
    httpd = Server(("0.0.0.0", port), RangeHandler)
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료했어요.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CBoE 검수 화면 로컬 서버")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--video", help="예: media/movie.mp4")
    ap.add_argument("--subs", help="예: subs/movie.srt (SDH 자막 바로 열기)")
    ap.add_argument("--work", help="CBoE 작업 폴더, 예: work/sample")
    ap.add_argument("--no-open", action="store_true", help="브라우저 자동 열기 끄기")
    a = ap.parse_args()
    url = player_url(a.port, a.video, a.subs, a.work)
    print(f"  ✦ CBoE 검수 화면 →  {url}\n  (종료: Ctrl+C)")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    start(a.port)
