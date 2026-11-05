"""논문 번역기 창 (paper-ko GUI).

python gui.py [PDF ...] [--auto] [--quit]
  PDF 를 주면 목록에 넣어 둔다. --auto: 바로 번역 시작, --quit: 끝나면 창 닫기.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

# exe(창 모드)에는 표준 출력이 없다 → print 가 죽지 않게
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import paper_ko as pk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    BaseTk = TkinterDnD.Tk
except Exception:  # 끌어다 놓기 없이도 동작
    DND_FILES, BaseTk = None, tk.Tk

APP_NAME = "논문 번역기"
SETTINGS = pk.DATA_DIR / "settings.json"
LOG_FILE = pk.DATA_DIR / "last_log.txt"
RES_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
INSTALL_CMD = "irm https://claude.ai/install.ps1 | iex"
MODELS = [("sonnet", "Sonnet — 기본, 빠름"), ("opus", "Opus — 더 정확, 느림"), ("haiku", "Haiku — 가장 빠름")]
NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def load_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


class LogWriter:
    """작업 스레드의 print 를 창의 기록 칸으로 보낸다. '\\r' 은 같은 줄 덮어쓰기."""

    def __init__(self, q, sink):
        self.q, self.sink = q, sink

    def write(self, s):
        if not s:
            return 0
        self.sink.append(s)
        for i, part in enumerate(s.split("\n")):
            if i:
                self.q.put(("nl", None))
            if "\r" in part:
                self.q.put(("cr", part.split("\r")[-1]))
            elif part:
                self.q.put(("text", part))
        return len(s)

    def flush(self):
        pass


class App:
    def __init__(self, root, files=(), auto=False, quit_when_done=False):
        self.root, self.auto, self.quit_when_done = root, auto, quit_when_done
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.results = []
        self.claude_exe = None
        self.st = load_settings()

        root.title(APP_NAME)
        self.k = k = max(root.winfo_fpixels("1i") / 96, 1.0)  # 화면 배율
        root.geometry(f"{int(780 * k)}x{int(820 * k)}")
        root.minsize(int(680 * k), int(660 * k))
        self.files = []  # 실제 경로 (목록에는 이름만 보임)
        try:
            root.iconbitmap(str(RES_DIR / "app.ico"))
        except Exception:
            pass
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        root.option_add("*Font", ("맑은 고딕", 10))
        style.configure(".", font=("맑은 고딕", 10))
        style.configure("Big.TButton", font=("맑은 고딕", 11, "bold"), padding=(18, 6))
        style.configure("Hint.TLabel", foreground="#666666")

        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)
        self._files(main)
        self._options(main)
        self._claude(main)
        self._run(main)

        for f in files:
            self.add_file(f)
        self.on_engine()
        threading.Thread(target=self.refresh_claude, daemon=True).start()
        root.after(100, self.pump)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if auto and files:
            root.after(1500, self.start)

    # ── 화면 구성 ──────────────────────────────────────────
    def _files(self, parent):
        box = ttk.LabelFrame(parent, text=" PDF 파일 — 여기로 끌어다 놓아도 됩니다 ", padding=8)
        box.pack(fill="both", expand=False)
        row = ttk.Frame(box)
        row.pack(fill="both", expand=True)
        self.lst = tk.Listbox(row, height=5, selectmode="extended", activestyle="none")
        sb = ttk.Scrollbar(row, orient="vertical", command=self.lst.yview)
        self.lst.configure(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        btns = ttk.Frame(box)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="PDF 추가…", command=self.pick_files).pack(side="left")
        ttk.Button(btns, text="선택 빼기", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Button(btns, text="모두 지우기", command=self.clear_files).pack(side="left")
        if DND_FILES:
            for w in (self.root, self.lst):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.on_drop)

    def _options(self, parent):
        box = ttk.LabelFrame(parent, text=" 번역 설정 ", padding=8)
        box.pack(fill="x", pady=(10, 0))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="번역 엔진").grid(row=0, column=0, sticky="w", pady=3)
        self.engine = tk.StringVar(value=self.st.get("engine", "claude"))
        f = ttk.Frame(box)
        f.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(f, text="Claude (구독, 품질 좋음)", value="claude", variable=self.engine,
                        command=self.on_engine).pack(side="left")
        ttk.Radiobutton(f, text="구글 무료 (품질 낮음, 많이 쓰면 막힘)", value="google", variable=self.engine,
                        command=self.on_engine).pack(side="left", padx=12)

        ttk.Label(box, text="Claude 모델").grid(row=1, column=0, sticky="w", pady=3)
        self.model = ttk.Combobox(box, state="readonly", width=28, values=[m[1] for m in MODELS])
        idx = next((i for i, m in enumerate(MODELS) if m[0] == self.st.get("model")), 0)
        self.model.current(idx)
        self.model.grid(row=1, column=1, sticky="w")

        ttk.Label(box, text="쪽 범위").grid(row=2, column=0, sticky="w", pady=3)
        f = ttk.Frame(box)
        f.grid(row=2, column=1, sticky="w")
        self.pages = ttk.Entry(f, width=16)
        self.pages.pack(side="left")
        ttk.Label(f, text="비우면 전체.  예: 1-5, 8   -3   10-", style="Hint.TLabel").pack(side="left", padx=8)

        self.dual = tk.BooleanVar(value=self.st.get("dual", True))
        ttk.Checkbutton(box, text="원문 | 번역 나란히 보는 PDF 도 만들기", variable=self.dual).grid(
            row=3, column=1, sticky="w", pady=3)

        ttk.Label(box, text="저장 위치").grid(row=4, column=0, sticky="w", pady=3)
        f = ttk.Frame(box)
        f.grid(row=4, column=1, sticky="ew")
        self.outmode = tk.StringVar(value=self.st.get("outmode", "beside"))
        ttk.Radiobutton(f, text="원본 PDF 옆", value="beside", variable=self.outmode).pack(side="left")
        ttk.Radiobutton(f, text="이 폴더:", value="dir", variable=self.outmode).pack(side="left", padx=(12, 4))
        self.outdir = ttk.Entry(f)
        self.outdir.insert(0, self.st.get("outdir", ""))
        self.outdir.pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="찾기…", command=self.pick_outdir).pack(side="left", padx=(4, 0))

    def _claude(self, parent):
        box = ttk.LabelFrame(parent, text=" Claude 연결 ", padding=8)
        box.pack(fill="x", pady=(10, 0))
        self.claude_box = box
        box.columnconfigure(1, weight=1)

        self.status = tk.Label(box, text="Claude Code 확인 중…", anchor="w", fg="#666666")
        self.status.grid(row=0, column=0, columnspan=3, sticky="ew")

        ttk.Label(box, text="구독 토큰 (선택)").grid(row=1, column=0, sticky="w", pady=(8, 3))
        self.token = ttk.Entry(box, show="•")
        self.token.grid(row=1, column=1, sticky="ew", pady=(8, 3), padx=(0, 6))
        self.show_tok = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="보기", variable=self.show_tok,
                        command=lambda: self.token.configure(show="" if self.show_tok.get() else "•")).grid(
            row=1, column=2, sticky="w", pady=(8, 3))

        hint = ("쓰는 순서: 이 칸의 토큰 → 프로그램 옆 .env 파일의 CLAUDE_CODE_OAUTH_TOKEN → 이 PC 의 Claude 로그인.  "
                "토큰은 'claude setup-token' 으로 받습니다.  이 칸에 넣은 토큰은 저장하지 않고, .env 는 직접 넣은 것만 읽습니다.  "
                "API 키는 읽지도 쓰지도 않으므로 요금이 나가지 않습니다.")
        ttk.Label(box, text=hint, style="Hint.TLabel", wraplength=int(700 * self.k), justify="left").grid(
            row=2, column=0, columnspan=3, sticky="w")

        f = ttk.Frame(box)
        f.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.btn_check = ttk.Button(f, text="연결 확인", command=self.check_connection)
        self.btn_check.pack(side="left")
        self.btn_login = ttk.Button(f, text="Claude 로그인", command=self.open_login)
        self.btn_login.pack(side="left", padx=6)
        self.btn_token = ttk.Button(f, text="토큰 받기", command=self.open_setup_token)
        self.btn_token.pack(side="left")
        ttk.Button(f, text=".env 열기", command=self.open_env).pack(side="left", padx=(6, 0))
        self.btn_install = ttk.Button(f, text="Claude Code 설치", command=self.install_claude)
        self.btn_install.pack(side="left", padx=6)
        ttk.Button(f, text="다시 찾기", command=lambda: threading.Thread(
            target=self.refresh_claude, daemon=True).start()).pack(side="left")

    def _run(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(12, 0))
        self.btn_start = ttk.Button(f, text="번역 시작", style="Big.TButton", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(f, text="취소", command=self.do_cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=8)
        self.prog = ttk.Progressbar(f, mode="determinate", maximum=1000)
        self.prog.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.prog_lbl = ttk.Label(parent, text="대기 중", style="Hint.TLabel")
        self.prog_lbl.pack(fill="x", pady=(4, 0))

        lf = ttk.Frame(parent)
        lf.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(lf, height=10, wrap="word", state="disabled", font=("맑은 고딕", 9),
                           background="#fafafa", relief="solid", borderwidth=1)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        r = ttk.Frame(parent)
        r.pack(fill="x", pady=(8, 0))
        self.btn_open = ttk.Button(r, text="번역본 열기", state="disabled",
                                   command=lambda: self.open_result(0))
        self.btn_open.pack(side="left")
        self.btn_dual = ttk.Button(r, text="나란히 보기 열기", state="disabled",
                                   command=lambda: self.open_result(1))
        self.btn_dual.pack(side="left", padx=6)
        self.btn_folder = ttk.Button(r, text="결과 폴더 열기", state="disabled", command=self.open_folder)
        self.btn_folder.pack(side="left")

    # ── 파일 목록 ──────────────────────────────────────────
    def add_file(self, f):
        p = Path(f)
        if p.is_dir():
            for q in sorted(p.glob("*.pdf")):
                self.add_file(q)
            return
        if p.suffix.lower() != ".pdf" or not p.exists():
            return
        if re.search(r"_ko(_dual)?(\(\d+\))?$", p.stem):  # 번역 결과물은 다시 넣지 않음
            return
        s = str(p.resolve())
        if s not in self.files:
            self.files.append(s)
            self.lst.insert("end", f"{p.name}      ({p.resolve().parent})")

    def clear_files(self):
        self.files.clear()
        self.lst.delete(0, "end")

    def on_drop(self, event):
        for f in self.root.tk.splitlist(event.data):
            self.add_file(f)

    def pick_files(self):
        for f in filedialog.askopenfilenames(title="번역할 PDF", filetypes=[("PDF", "*.pdf")]):
            self.add_file(f)

    def remove_selected(self):
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)
            del self.files[i]

    def pick_outdir(self):
        d = filedialog.askdirectory(title="결과를 저장할 폴더")
        if d:
            self.outdir.delete(0, "end")
            self.outdir.insert(0, d)
            self.outmode.set("dir")

    # ── Claude ────────────────────────────────────────────
    def on_engine(self):
        st = "normal" if self.engine.get() == "claude" else "disabled"
        self.model.configure(state="readonly" if st == "normal" else "disabled")
        self.token.configure(state=st)

    def set_status(self, text, color):
        self.root.after(0, lambda: self.status.configure(text=text, fg=color))

    def refresh_claude(self):
        exe = pk.find_claude()
        self.claude_exe = exe
        if not exe:
            self.set_status("● Claude Code 가 이 PC 에 없습니다 — 'Claude Code 설치' 를 누르거나 구글 무료 엔진을 쓰세요",
                            "#c0392b")
            return
        if pk.env_file_token():
            self.set_status(f"● .env 의 토큰을 씁니다 ({pk.env_file_path()}) — [연결 확인]으로 맞는지 확인하세요",
                            "#1f618d")
            return
        info = pk.claude_status(exe)
        if info and info.get("loggedIn"):
            plan = info.get("subscriptionType") or info.get("authMethod") or ""
            self.set_status(f"● Claude Code 준비됨 — 이 PC 에 로그인되어 있습니다 ({plan})", "#1e8449")
        else:
            self.set_status("● Claude Code 는 있지만 로그인이 안 되어 있습니다 — 'Claude 로그인' 을 누르거나 토큰을 넣으세요",
                            "#b9770e")

    def check_connection(self):
        if not self.claude_exe:
            messagebox.showwarning(APP_NAME, "Claude Code 가 설치되어 있지 않습니다.")
            return
        tok = self.token.get().strip() or None
        src = "창에 넣은 토큰"
        if not tok:
            tok = pk.env_file_token()
            src = ".env 토큰" if tok else "이 PC 로그인"
        self.btn_check.configure(state="disabled", text="확인 중…")

        def work():
            ok, msg = pk.check_claude(self.claude_exe, tok)

            def show():
                self.btn_check.configure(state="normal", text="연결 확인")
                if ok:
                    messagebox.showinfo(APP_NAME, f"연결됨 ({src}). 번역할 수 있습니다.")
                    if tok:
                        self.status.configure(text=f"● {src}으로 연결됨", fg="#1e8449")
                else:
                    messagebox.showerror(APP_NAME, f"연결 실패 ({src})\n\n{msg}")
            self.root.after(0, show)
        threading.Thread(target=work, daemon=True).start()

    def _console(self, args, title):
        try:
            subprocess.Popen(["cmd", "/k", "title", title, "&&"] + args, creationflags=NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    def open_login(self):
        if not self.claude_exe:
            messagebox.showwarning(APP_NAME, "먼저 Claude Code 를 설치하세요.")
            return
        messagebox.showinfo(APP_NAME, "새 창에서 Anthropic 공식 로그인을 엽니다.\n브라우저에서 로그인을 마친 뒤 "
                                      "그 창을 닫고 '다시 찾기' 를 누르세요.")
        self._console([self.claude_exe, "auth", "login"], "Claude 로그인")

    def open_setup_token(self):
        if not self.claude_exe:
            messagebox.showwarning(APP_NAME, "먼저 Claude Code 를 설치하세요.")
            return
        messagebox.showinfo(APP_NAME, "새 창에서 'claude setup-token' 을 실행합니다.\n브라우저에서 승인하면 창에 토큰이 "
                                      "나옵니다. 그 토큰을 복사해서\n"
                                      "  · [.env 열기] 로 연 파일의 CLAUDE_CODE_OAUTH_TOKEN= 뒤에 붙여넣고 저장하거나 (계속 사용)\n"
                                      "  · '구독 토큰' 칸에 붙여넣으세요 (이번에만).\n\n"
                                      "토큰은 계정 비밀번호와 같으니 다른 사람에게 주지 마세요.")
        self._console([self.claude_exe, "setup-token"], "Claude 토큰 받기")

    def open_env(self):
        """프로그램 옆 .env 를 메모장으로 연다. 없으면 빈 양식을 만든다."""
        p = pk.env_file_path()
        try:
            if not p.exists():
                p.write_text(pk.ENV_TEMPLATE, encoding="utf-8")
            proc = subprocess.Popen(["notepad", str(p)])
        except Exception as e:
            messagebox.showerror(APP_NAME, f".env 를 열지 못했습니다: {p}\n{e}")
            return

        def wait():  # 메모장을 닫으면 상태 다시 확인
            proc.wait()
            self.refresh_claude()
        threading.Thread(target=wait, daemon=True).start()

    def install_claude(self):
        if not messagebox.askyesno(APP_NAME, "Anthropic 공식 설치 명령을 새 PowerShell 창에서 실행합니다.\n\n"
                                             f"    {INSTALL_CMD}\n\n설치가 끝나면 '다시 찾기' 를 누르세요. 계속할까요?"):
            return
        try:
            subprocess.Popen(["powershell", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass",
                              "-Command", INSTALL_CMD], creationflags=NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    # ── 실행 ──────────────────────────────────────────────
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        files = list(self.files)
        if not files:
            messagebox.showinfo(APP_NAME, "번역할 PDF 를 추가하세요.")
            return
        engine = self.engine.get()
        if engine == "claude" and not self.claude_exe:
            messagebox.showwarning(APP_NAME, "Claude Code 가 없습니다. 설치하거나 '구글 무료' 엔진을 고르세요.")
            return
        pages = self.pages.get().strip()
        if pages and not re.fullmatch(r"[\d\s,\-]+", pages):
            messagebox.showwarning(APP_NAME, "쪽 범위는 숫자와 - , 만 쓸 수 있습니다. 예: 1-5, 8")
            return
        out = None
        if self.outmode.get() == "dir":
            out = self.outdir.get().strip()
            if not out:
                messagebox.showwarning(APP_NAME, "저장할 폴더를 고르세요.")
                return
        model = MODELS[self.model.current()][0]
        save_settings({"engine": engine, "model": model, "dual": self.dual.get(),
                       "outmode": self.outmode.get(), "outdir": self.outdir.get().strip()})

        self.cancel.clear()
        self.results = []
        self._busy(True)
        self.log_clear()
        args = SimpleNamespace(engine=engine, model=model, pages=pages or None, out=out,
                               no_dual=not self.dual.get(), debug=False, cancel=self.cancel,
                               token=self.token.get().strip() or None if engine == "claude" else None,
                               claude_exe=self.claude_exe, progress=None)
        self.worker = threading.Thread(target=self.run_jobs, args=(files, args), daemon=True)
        self.worker.start()

    def run_jobs(self, files, args):
        old = sys.stdout, sys.stderr
        sink = []
        sys.stdout = sys.stderr = LogWriter(self.q, sink)
        results, n = [], len(files)
        try:
            for i, f in enumerate(files):
                name = Path(f).name
                args.progress = (lambda frac, text, i=i, name=name:
                                 self.q.put(("prog", (i + frac) / n, f"[{i + 1}/{n}] {name} — {text}")))
                try:
                    r = pk.translate_pdf(f, args)
                    if r:
                        results.append(r)
                except pk.Cancelled:
                    print("\n■ 취소했습니다. 그때까지 번역한 문단은 저장돼 있어 다시 돌리면 이어서 합니다.")
                    break
                except Exception as e:
                    print(f"\n✖ 실패: {name}: {e}")
        finally:
            sys.stdout, sys.stderr = old
            try:
                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                LOG_FILE.write_text("".join(sink), encoding="utf-8")
            except Exception:
                pass
            self.q.put(("done", results))

    def do_cancel(self):
        self.cancel.set()
        self.btn_cancel.configure(state="disabled")
        self.prog_lbl.configure(text="취소하는 중… (진행 중인 요청을 멈춥니다)")

    def _busy(self, busy):
        self.btn_start.configure(state="disabled" if busy else "normal")
        self.btn_cancel.configure(state="normal" if busy else "disabled")
        for b in (self.btn_open, self.btn_dual, self.btn_folder):
            b.configure(state="disabled")

    def pump(self):
        try:
            while True:
                kind, val, *rest = self.q.get_nowait()
                if kind in ("text", "cr", "nl"):
                    self.log_put(kind, val)
                elif kind == "prog":
                    self.prog["value"] = int(val * 1000)
                    self.prog_lbl.configure(text=rest[0])
                elif kind == "done":
                    self.finish(val)
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def finish(self, results):
        self.results = results
        self._busy(False)
        if results:
            self.prog["value"] = 1000
            self.prog_lbl.configure(text=f"완료 — {len(results)}개 번역")
            self.btn_open.configure(state="normal")
            self.btn_folder.configure(state="normal")
            if results[-1][1]:
                self.btn_dual.configure(state="normal")
        elif not self.cancel.is_set():
            self.prog_lbl.configure(text="끝났지만 만든 파일이 없습니다 — 기록을 확인하세요")
        if self.quit_when_done:
            self.root.after(500, self.root.destroy)

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def log_put(self, kind, val):
        self.log.configure(state="normal")
        if kind == "nl":
            self.log.insert("end-1c", "\n")
        elif kind == "cr":
            self.log.delete("end-1c linestart", "end-1c")
            self.log.insert("end-1c", val)
        else:
            self.log.insert("end-1c", val)
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_result(self, which):
        if self.results and self.results[-1][which]:
            os.startfile(self.results[-1][which])

    def open_folder(self):
        if self.results:
            os.startfile(Path(self.results[-1][0]).parent)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(APP_NAME, "번역 중입니다. 멈추고 닫을까요?"):
                return
            self.cancel.set()
        self.root.destroy()


def dpi_aware():
    """고해상도 화면에서 글자가 흐리지 않게 (창 만들기 전에 불러야 함)."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    argv = sys.argv[1:]
    auto, quit_ = "--auto" in argv, "--quit" in argv
    files = [a for a in argv if not a.startswith("--")]
    dpi_aware()
    root = BaseTk()
    App(root, files, auto, quit_)
    root.mainloop()


if __name__ == "__main__":
    main()
