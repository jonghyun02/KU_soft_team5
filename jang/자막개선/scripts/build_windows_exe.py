"""Compile the native Windows GUI executable `MoS-SDH-Studio.exe` (zero console window, custom .ico)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from mos_sdh.config import get_default_cache_dir  # noqa: E402
from mos_sdh.desktop.theme import ensure_studio_ico  # noqa: E402

CSHARP_LAUNCHER_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("MoS-SDH Studio")]
[assembly: AssemblyDescription("5-Stage Audio-Visual Manner of Speech Barrier-Free Subtitle Studio")]
[assembly: AssemblyCompany("MoS-SDH Project")]
[assembly: AssemblyProduct("MoS-SDH Studio v1.0")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class Program
{
    private static string QuoteArg(string arg)
    {
        if (string.IsNullOrEmpty(arg))
            return "\"\"";
        if (arg.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0)
            return arg;
        return "\"" + arg.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    private static string ResolvePythonW(string baseDir)
    {
        var candidates = new List<string>();
        string envVenv = Environment.GetEnvironmentVariable("UV_PROJECT_ENVIRONMENT");
        if (!string.IsNullOrWhiteSpace(envVenv))
            candidates.Add(Path.Combine(envVenv, "Scripts", "pythonw.exe"));

        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        candidates.Add(Path.Combine(localAppData, "mos-sdh", "venv", "Scripts", "pythonw.exe"));
        candidates.Add(Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe"));

        foreach (string c in candidates)
        {
            if (File.Exists(c))
                return c;
        }
        return null;
    }

    [STAThread]
    private static int Main(string[] args)
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
        string pythonw = ResolvePythonW(baseDir);

        if (pythonw == null)
        {
            MessageBox.Show(
                "MoS-SDH Studio 가상환경(pythonw.exe)을 찾을 수 없습니다.\n\n" +
                "경로 확인: %LOCALAPPDATA%\\mos-sdh\\venv\\Scripts\\pythonw.exe\n" +
                "프로젝트 폴더에서 'uv sync'를 먼저 실행해 주세요.",
                "MoS-SDH Studio 실행 오류",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }

        bool waitForExit = false;
        var argBuilder = new StringBuilder("-m mos_sdh.desktop");
        foreach (string a in args)
        {
            if (a == "--wait" || a == "--smoke-exit-ms" || a.StartsWith("--smoke-exit-ms="))
                waitForExit = true;
            if (a != "--wait")
            {
                argBuilder.Append(' ');
                argBuilder.Append(QuoteArg(a));
            }
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonw,
            Arguments = argBuilder.ToString(),
            WorkingDirectory = baseDir,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        string srcDir = Path.Combine(baseDir, "src");
        string existingPyPath = Environment.GetEnvironmentVariable("PYTHONPATH");
        psi.EnvironmentVariables["PYTHONPATH"] = string.IsNullOrEmpty(existingPyPath)
            ? srcDir
            : (srcDir + ";" + existingPyPath);
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";

        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("MOS_SDH_CACHE_DIR")))
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            psi.EnvironmentVariables["MOS_SDH_CACHE_DIR"] = Path.Combine(localAppData, "mos-sdh", "cache");
        }

        try
        {
            Process proc = Process.Start(psi);
            if (proc == null)
                return 1;
            if (waitForExit)
            {
                proc.WaitForExit();
                return proc.ExitCode;
            }
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "MoS-SDH Studio 구동 중 오류가 발생했습니다:\n" + ex.Message,
                "MoS-SDH Studio 오류",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }
}
"""


def build_exe() -> Path:
    ico_path = ensure_studio_ico(ROOT_DIR / "assets" / "mos_sdh.ico")
    build_dir = get_default_cache_dir().parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cs_path = build_dir / "MoSStudioLauncher.cs"
    cs_path.write_text(CSHARP_LAUNCHER_SOURCE.strip(), encoding="utf-8-sig")

    csc_candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
    ]
    csc = next((p for p in csc_candidates if p.exists()), None)
    if csc is None:
        raise FileNotFoundError("Windows C# compiler (csc.exe) not found under C:\\Windows\\Microsoft.NET")

    exe_out = ROOT_DIR / "MoS-SDH-Studio.exe"
    cmd = [
        str(csc),
        "/nologo",
        "/target:winexe",
        "/platform:x64",
        "/optimize+",
        f"/win32icon:{ico_path}",
        f"/out:{exe_out}",
        "/r:System.Windows.Forms.dll",
        str(cs_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"csc.exe compilation failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    print(f"[OK] Icon generated : {ico_path} ({ico_path.stat().st_size:,} bytes)")
    print(f"[OK] Native WinExe  : {exe_out} ({exe_out.stat().st_size:,} bytes)")
    return exe_out


if __name__ == "__main__":
    build_exe()
