# OneDrive 동기화 충돌 방지를 위해 로컬 AppData 경로의 가상환경 및 캐시를 자동 바인딩하는 래퍼 스크립트
$ErrorActionPreference = "Stop"
if (-not $env:UV_PROJECT_ENVIRONMENT) {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA "mos-sdh\venv"
}
if (-not $env:MOS_SDH_CACHE_DIR) {
    $env:MOS_SDH_CACHE_DIR = Join-Path $env:LOCALAPPDATA "mos-sdh\cache"
}
uv run mos-sdh @args
