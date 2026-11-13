"""한컴오피스 자동화 일꾼(별도 프로세스).

부모(convert.py)가 `python -I -m sinkprep.hancom_worker` 로 띄운다. 통신은 JSON 줄.

stdin  한 줄 = 일 하나: {"id": "...", "src": "<복사본 경로>", "fmt": "hwp5", "out": "<pdf 경로>"}
       {"cmd": "quit"} 또는 EOF 면 한글을 닫고 끝낸다.
stdout {"event": "ready", "hwp_pid": 123, "version": "13, 0, 0, 711", "module": true}
       {"event": "done", "id": "...", "ok": true, "how": "open|insert", "pages": 53,
        "error": null, "seconds": 3.2, "killed_hword": 0}

안전 장치
- 원본은 신뢰할 수 없다. 부모가 원본을 cache/convert_work/ 에 ASCII 이름으로 복사해서 넘긴다.
  일꾼은 원본 폴더를 건드리지 않는다.
- 창: 자동화로 띄운 한글은 창이 숨어 있고(Visible=False 를 다시 지정), 대화상자는
  SetMessageBoxMode 로 '아니오/취소/확인'을 자동으로 고른다(실행을 묻는 질문에 '예'를 하지 않음).
  첫 Open 이 실패한 문서만 '예/아니오/취소' 상자에 '예'로 한 번 더 연다(open_yes, MSGBOX_YES 설명 참고).
- doc/rtf 는 Open 을 쓰지 않는다. 한글 2024 의 Open 은 doc/rtf 를 한워드(Hword.exe) 창으로
  넘겨 화면에 띄운다(실험으로 확인). 대신 빈 문서에 Insert 로 끼워 넣는다.
- 혹시 한워드가 이 한글의 자식으로 뜨면 바로 죽인다.
- 매크로: HWP5 문서 스크립트는 부모가 미리 검사해서 기본 틀이 아니면 한컴으로 보내지 않는다.
"""
from __future__ import annotations

import json
import os
import sys
import time
import winreg

HWP_PROGID = "HWPFrame.HwpObject"
MODULE_NAME = "FilePathCheckerModuleExample"
MODULE_KEY = r"SOFTWARE\HNC\HwpAutomation\Modules"
DEFAULT_DLL = r"C:\Program Files (x86)\Hnc\securitymoduel\FilePathCheckerModuleExample.dll"

# SetMessageBoxMode: 상자 종류마다 누를 단추 하나.
#   MB_OK 확인(0x1) | MB_OKCANCEL 취소(0x20) | MB_ABORTRETRYIGNORE 중단(0x100)
#   | MB_YESNOCANCEL 취소(0x4000) | MB_YESNO 아니오(0x20000) | MB_RETRYCANCEL 취소(0x200000)
MSGBOX_MODE = 0x1 | 0x20 | 0x100 | 0x4000 | 0x20000 | 0x200000
# 두 번째 시도: '예/아니오/취소' 상자만 '예'(0x1000). 마린보이 [55927](HWP 5.0.2.4)은 열 때 이 상자가
# 떠서 '취소'면 Open 이 False, '예'면 76쪽으로 열린다(실험으로 확인, 상자 글은 화면에 띄우지 않아 미확인).
# 매크로 실행 질문에 '예'가 될 수 있으므로 부모가 HWP5 스크립트를 미리 검사한 문서에만 쓴다(첫 시도 실패 시).
MSGBOX_YES = MSGBOX_MODE - 0x4000 + 0x1000

OPEN_ARG = "forceopen:true;suspendpassword:true;versionwarning:false;lock:false"

# 형식별 (방법, 한글 형식 이름). 실험 결과(2026-11-13, 한글 2024 13.0.0.711):
#   hwp3: Open(fmt="HWP") 는 실패, 자동 판별("")은 성공.
#   hwp5: "HWP", docx: "OOXML" 성공("MSWORDX" 실패).
#   doc: Open 은 한워드 창으로 넘어감 -> Insert("MSWORD") 로 성공.
#   rtf: Open 은 한워드 창으로 넘어감 -> Insert("RTF").
# open_yes = 같은 Open 을 MSGBOX_YES 로 다시.
PLAN = {
    "hwp3": [("open", ""), ("open_yes", ""), ("insert", "")],
    "hwp5": [("open", "HWP"), ("open_yes", "HWP"), ("insert", "HWP")],
    "hwpx": [("open", "HWPX"), ("open_yes", "HWPX"), ("insert", "HWPX")],
    "docx": [("open", "OOXML"), ("open_yes", "OOXML"), ("insert", "OOXML")],
    "doc": [("insert", "MSWORD")],
    "rtf": [("insert", "RTF")],
}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def ensure_security_module(dll: str = DEFAULT_DLL) -> bool:
    """HKCU\\SOFTWARE\\HNC\\HwpAutomation\\Modules 에 보안 모듈 경로를 둔다(없거나 다르면 씀)."""
    if not os.path.exists(dll):
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MODULE_KEY) as k:
            cur, _ = winreg.QueryValueEx(k, MODULE_NAME)
            if cur == dll:
                return True
    except OSError:
        pass
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, MODULE_KEY) as k:
        winreg.SetValueEx(k, MODULE_NAME, 0, winreg.REG_SZ, dll)
    return True


def _hword_children(hwp_pid: int) -> list:
    import psutil
    out = []
    for p in psutil.process_iter(["name", "ppid"]):
        if (p.info["name"] or "").lower() == "hword.exe" and p.info["ppid"] == hwp_pid:
            out.append(p)
    return out


def kill_hword_children(hwp_pid: int) -> int:
    n = 0
    for p in _hword_children(hwp_pid):
        try:
            p.kill()
            n += 1
        except Exception:
            pass
    return n


def visible_windows(pid: int) -> list[str]:
    """이 프로세스의 화면 안 보이는 최상위 창 제목(최소화 -32000·숨김 -16000 같은 화면 밖 좌표 제외)."""
    import win32gui
    import win32process

    out: list[str] = []

    def cb(h, _):
        try:
            _, wpid = win32process.GetWindowThreadProcessId(h)
            if wpid == pid and win32gui.IsWindowVisible(h):
                r = win32gui.GetWindowRect(h)
                if r[2] > -10000 and r[3] > -10000 and r[2] - r[0] > 0 and r[3] - r[1] > 0:
                    out.append(win32gui.GetWindowText(h)[:80])
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass
    return out


def hide_windows(pid: int) -> None:
    import win32con
    import win32gui
    import win32process

    def cb(h, _):
        try:
            _, wpid = win32process.GetWindowThreadProcessId(h)
            if wpid == pid and win32gui.IsWindowVisible(h):
                win32gui.ShowWindow(h, win32con.SW_HIDE)
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass


class Hancom:
    def __init__(self) -> None:
        import pythoncom
        import win32com.client
        import win32process

        pythoncom.CoInitialize()
        self.module_reg = ensure_security_module()
        self.hwp = win32com.client.DispatchEx(HWP_PROGID)
        win = self.hwp.XHwpWindows.Item(0)
        win.Visible = False
        self.pid = int(win32process.GetWindowThreadProcessId(int(win.WindowHandle))[1])
        hide_windows(self.pid)
        self.module = bool(self.hwp.RegisterModule("FilePathCheckDLL", MODULE_NAME))
        self.hwp.SetMessageBoxMode(MSGBOX_MODE)
        self.version = str(self.hwp.Version)

    def convert(self, src: str, fmt: str, out: str) -> dict:
        hwp = self.hwp
        t0 = time.time()
        errors: list[str] = []
        how = None
        pages = None
        for method, name in PLAN.get(fmt, []):
            try:
                hwp.Clear(1)
                hwp.SetMessageBoxMode(MSGBOX_YES if method == "open_yes" else MSGBOX_MODE)
                if method in ("open", "open_yes"):
                    ok = bool(hwp.Open(src, name, OPEN_ARG))
                else:
                    r = hwp.Insert(src, name, "")
                    ok = r is None or bool(r)
            except Exception as e:  # COM 오류
                ok = False
                errors.append(f"{method}:{name or 'auto'}:{e}")
            finally:
                try:
                    hwp.SetMessageBoxMode(MSGBOX_MODE)
                except Exception:
                    pass
            killed = kill_hword_children(self.pid)
            if killed:
                ok = False
                errors.append(f"{method}:{name or 'auto'}:hword_spawned")
            if not ok:
                if not errors or not errors[-1].startswith(f"{method}:"):
                    errors.append(f"{method}:{name or 'auto'}:returned_false")
                continue
            try:
                pages = int(hwp.PageCount)
            except Exception:
                pages = None
            try:
                if os.path.exists(out):
                    os.remove(out)
                saved = bool(hwp.SaveAs(out, "PDF", ""))
            except Exception as e:
                saved = False
                errors.append(f"saveas:{e}")
            if saved and os.path.exists(out) and os.path.getsize(out) > 0:
                how = method
                break
            errors.append(f"{method}:{name or 'auto'}:saveas_failed")
        try:
            hwp.Clear(1)
        except Exception:
            pass
        killed = kill_hword_children(self.pid)
        wins = visible_windows(self.pid)
        if wins:
            hide_windows(self.pid)
        return {"ok": how is not None, "how": how, "pages": pages,
                "error": None if how else ("; ".join(errors) or f"no_plan:{fmt}"),
                "seconds": round(time.time() - t0, 3), "killed_hword": killed,
                "windows": wins}

    def quit(self) -> None:
        try:
            self.hwp.Clear(1)
        except Exception:
            pass
        try:
            self.hwp.Quit()
        except Exception:
            pass


def main() -> int:
    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    try:
        hc = Hancom()
    except Exception as e:
        emit({"event": "error", "error": f"start:{e}"})
        return 2
    emit({"event": "ready", "pid": os.getpid(), "hwp_pid": hc.pid, "version": hc.version,
          "module": hc.module, "module_reg": hc.module_reg})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            if job.get("cmd") == "quit":
                break
            res = hc.convert(job["src"], job["fmt"], job["out"])
            res.update(event="done", id=job.get("id"))
            emit(res)
    finally:
        hc.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
