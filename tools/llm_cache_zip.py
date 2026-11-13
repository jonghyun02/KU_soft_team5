"""LLM 응답 캐시(cache/llm) 묶음: 저장소에 넣는 data/llm_cache.zip 을 만들거나 푼다.

claude 응답은 같은 입력에도 매번 조금 다를 수 있어서, 캐시를 잃으면 다시 불러도 같은 결과가 나온다는 보장이 없다.
그래서 캐시를 저장소에 묶어 둔다(다른 PC 에서도 `sinkprep all` 이 claude 를 부르지 않고 같은 결과를 낸다).

pack   cache/llm/*.json -> data/llm_cache.zip. 같은 캐시면 같은 바이트(파일 차례·시각 고정). 계정 메타데이터
       (session_id, uuid, credential_expires_at)는 뺀다. 결과에 쓰는 칸(response, 키, 모델, 프롬프트 버전)은 그대로.
unpack data/llm_cache.zip -> cache/llm. 이름이 'llm/{40자 16진}.json' 인 항목만, 없는 파일만 쓴다(있으면 건너뜀).
       JSON 의 key 가 파일 이름과 같고 schema 가 맞는 것만.

사용: .venv/Scripts/python -I tools/llm_cache_zip.py pack|unpack
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import llm, paths, util  # noqa: E402

ZIP = ROOT / "data" / "llm_cache.zip"
PRIVATE = ("session_id", "uuid", "credential_expires_at")
RX_NAME = re.compile(r"^llm/([0-9a-f]{40})\.json$")
STAMP = (2026, 1, 1, 0, 0, 0)


def pack() -> int:
    files = sorted(llm.cache_dir().glob("*.json"))
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    tmp = ZIP.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            rec = util.read_json(f)
            env = rec.get("envelope")
            if isinstance(env, dict):
                rec["envelope"] = {k: v for k, v in env.items() if k not in PRIVATE}
            data = (json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
            zi = zipfile.ZipInfo(f"llm/{f.name}", date_time=STAMP)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    if ZIP.exists() and ZIP.read_bytes() == tmp.read_bytes():
        tmp.unlink()
        print(f"pack: 바뀐 것 없음 ({len(files)}개, {ZIP.relative_to(ROOT)})")
        return 0
    tmp.replace(ZIP)
    print(f"pack: {len(files)}개 -> {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size:,} 바이트)")
    return 0


def unpack() -> int:
    if not ZIP.exists():
        print(f"unpack: {ZIP.relative_to(ROOT)} 없음")
        return 1
    out = llm.cache_dir()
    paths.assert_not_raw(out)
    out.mkdir(parents=True, exist_ok=True)
    wrote = skipped = bad = 0
    with zipfile.ZipFile(ZIP) as z:
        for zi in z.infolist():
            m = RX_NAME.match(zi.filename)
            if not m:
                bad += 1
                continue
            dst = out / f"{m.group(1)}.json"
            if dst.exists():
                skipped += 1
                continue
            try:
                rec = json.loads(z.read(zi).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                bad += 1
                continue
            if rec.get("schema") != llm.CACHE_SCHEMA or rec.get("key") != m.group(1):
                bad += 1
                continue
            util.write_json(dst, rec)
            wrote += 1
    print(f"unpack: 새로 씀 {wrote}, 이미 있음 {skipped}, 건너뛴 이상한 항목 {bad} -> {out.relative_to(ROOT)}")
    return 1 if bad else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pack":
        return pack()
    if cmd == "unpack":
        return unpack()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
