"""매니페스트에 phonemes 칸을 더하고 characters.txt 를 쓴다(명세 3.12).

phonemes = "".join(pstts.paper.g2p.phonemes(text, lang)): 빈 토큰 없이 이어 붙인 음소 글(영어 phonemizer espeak, 기본
en-gb 강세표 없음, 한국어 g2pK + 호환 자모). 토큰이 모두 글자 하나라 list(phonemes) 가 그대로 토큰 열이다. 학습 때 Coqui
TTSTokenizer 가 글자마다 id 로 바꾸고 빈 토큰을 끼운 열은 추론 때 pstts.paper.yourtts 가 g2p.tokens(text, lang) 를 id 로
바꾼 열과 같다. 곧 학습과 추론이 같은 G2P 를 쓴다. phonemes 가 비는 발화는 빼고 센다.
G2P 설정(영어 목소리, 강세표, 토큰 꼴의 판)은 g2p.json 에 적는다. train_yourtts.py 가 학습 폴더로 복사하고, 추론
(pstts.paper.yourtts)이 PaperConfig 와 같은지 확인한다. 영어 설정은 --en-voice, --en-stress 로 바꾼다(기본은 논문 판 기본).

characters.txt: 세 매니페스트에 나온 토큰 전부. 줄마다 'U+코드<TAB>글자<TAB>언어=종류:수 ...'. 첫 칸이 기준이다(공백 토큰이
줄 끝 공백 정리에 흔들리지 않게). 코드포인트 순이고 이 순서가 모델 글자 id 순서다(train_yourtts.py).
--extra-en, --extra-ko(한 줄 한 문장)를 주면 그 글의 토큰도 글자 목록에만 더한다(학습 줄은 아님. 'extra_언어=종류:수' 로
적힌다). 학습에 안 나온 토큰의 임베딩은 학습되지 않으므로 기본은 쓰지 않는다. smoke_train.py 는 점검용 모델이 다른 문장도
읽을 수 있게 쓴다.

실행 예: .venv\\Scripts\\python.exe scripts\\yourtts\\phonemize_manifest.py --manifest-dir D:\\data\\pstts_tts
g2pK 는 문장 하나에 약 9 ms(이 PC 2,000줄 잰 값)라 한국어 10만 줄이면 15분쯤 걸린다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_manifest import MANIFEST_COLUMNS, SPLITS, manifest_path, read_manifest, write_manifest  # noqa: E402

from pstts.paper import g2p  # noqa: E402

PHONEMES = "phonemes"
COLUMNS = (*MANIFEST_COLUMNS, PHONEMES)
CHARACTERS_NAME = "characters.txt"


def phonemize_text(text: str, lang: str, en_voice: str = g2p.EN_VOICE, en_stress: bool = g2p.EN_WITH_STRESS) -> str:
    """빈 토큰 없이 이어 붙인 음소 글."""
    return "".join(g2p.phonemes(text, lang, en_voice=en_voice, en_stress=en_stress))


def write_characters(path: str | os.PathLike, counts: dict[str, Counter]) -> list[str]:
    """characters.txt 를 쓰고 토큰 목록(코드포인트 순)을 돌려준다.

    counts = {출처: Counter(토큰)}. 출처는 'en', 'ko'(학습 줄) 또는 'extra_en', 'extra_ko'(글자 목록에만 더한 글).
    """
    tokens = sorted({t for c in counts.values() for t in c}, key=ord)
    lines = ["# 논문 판 Your-TTS 글자 목록. 줄마다 U+코드<TAB>글자<TAB>출처=종류:수. 첫 칸이 기준이고 이 순서가 id 순서다."]
    for t in tokens:
        if len(t) != 1:
            raise ValueError(f"토큰은 글자 하나여야 합니다: {t!r}")
        kinds = " ".join(f"{src}={g2p.token_kind(t, src.rsplit('_', 1)[-1])}:{c[t]}"
                         for src, c in sorted(counts.items()) if t in c)
        lines.append(f"U+{ord(t):04X}\t{t if t.isprintable() and not t.isspace() else ''}\t{kinds}")
    p = Path(path)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    os.replace(tmp, p)
    return tokens


def read_characters(path: str | os.PathLike) -> list[str]:
    """characters.txt 의 토큰 목록(파일 순서). 첫 칸 U+코드만 본다."""
    out: list[str] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        code = line.split("\t", 1)[0].strip()
        if not code.startswith("U+"):
            raise ValueError(f"{path}:{i} 줄 첫 칸이 U+코드가 아닙니다: {line!r}")
        out.append(chr(int(code[2:], 16)))
    if len(set(out)) != len(out):
        raise ValueError(f"{path} 에 같은 토큰이 두 번 있습니다.")
    if not out:
        raise ValueError(f"{path} 에 토큰이 없습니다.")
    return out


def phonemize_manifests(folder: Path, progress_every: int = 2000, extra: dict[str, list[str]] | None = None,
                        en_voice: str = g2p.EN_VOICE, en_stress: bool = g2p.EN_WITH_STRESS) -> dict:
    """세 매니페스트에 phonemes 칸을 채워 다시 쓰고 characters.txt 와 g2p.json 을 쓴다. 요약 dict 를 돌려준다.

    extra = {언어: 문장들}: 그 토큰을 글자 목록에만 더한다(학습 줄은 아님).
    """
    voice = g2p.check_en_voice(en_voice)
    counts: dict[str, Counter] = {"en": Counter(), "ko": Counter()}
    summary: dict = {"files": {}, "dropped_empty": {}, "seconds": 0.0}
    start = time.perf_counter()
    done = 0
    for split in SPLITS:
        path = manifest_path(folder, split)
        rows = read_manifest(path)
        kept: list[dict[str, str]] = []
        dropped = 0
        for row in rows:
            lang = g2p.check_lang(row["lang"])
            ph = phonemize_text(row["text"], lang, voice, en_stress)
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  음소화 {done}줄 ({time.perf_counter() - start:.0f}초)", file=sys.stderr, flush=True)
            if not ph:
                dropped += 1
                continue
            counts[lang].update(ph)
            kept.append({**{c: row[c] for c in MANIFEST_COLUMNS}, PHONEMES: ph})
        write_manifest(path, kept, COLUMNS)
        summary["files"][split] = {"path": str(path.resolve()), "rows": len(kept)}
        summary["dropped_empty"][split] = dropped
    trained = {t for c in counts.values() for t in c}
    for lang, texts in (extra or {}).items():
        code = g2p.check_lang(lang)
        extra_counts = counts.setdefault(f"extra_{code}", Counter())
        for text in texts:
            extra_counts.update(phonemize_text(text, code, voice, en_stress))
        summary.setdefault("extra_sentences", {})[code] = len(texts)
    counts = {k: v for k, v in counts.items() if v}
    tokens = write_characters(folder / CHARACTERS_NAME, counts)
    summary["characters"] = {"path": str((folder / CHARACTERS_NAME).resolve()), "count": len(tokens),
                             "from_training_rows": len(trained), "only_from_extra": len(set(tokens) - trained)}
    summary["unknown_tokens"] = sorted(f"{lang}:{tok}" for lang, tok in g2p.UNKNOWN_TOKENS)
    summary["g2p_versions"] = g2p.versions()
    settings = g2p.write_settings(folder / g2p.SETTINGS_NAME, voice, en_stress)
    summary["g2p"] = {"path": str((folder / g2p.SETTINGS_NAME).resolve()), "scheme": settings["scheme"],
                      "en": settings["en"]}
    summary["seconds"] = round(time.perf_counter() - start, 2)
    (folder / "phonemize_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="매니페스트에 phonemes 칸을 더하고 characters.txt 를 쓴다.")
    ap.add_argument("--manifest-dir", type=Path, required=True, help="manifest_{train,valid,test}.csv 가 있는 폴더")
    ap.add_argument("--extra-en", type=Path, help="글자 목록에만 더할 영어 문장 파일(한 줄 한 문장, 기본 없음)")
    ap.add_argument("--extra-ko", type=Path, help="글자 목록에만 더할 한국어 문장 파일(한 줄 한 문장, 기본 없음)")
    ap.add_argument("--en-voice", default=g2p.EN_VOICE,
                    help=f"영어 espeak 목소리(기본 {g2p.EN_VOICE}: 논문 Fig. 5c 의 영어 모음과 같음)")
    ap.add_argument("--en-stress", action="store_true", help="영어 강세표를 토큰에 남긴다(기본 끔: 논문 Fig. 4)")
    args = ap.parse_args(argv)
    extra = {lang: [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
             for lang, p in (("en", args.extra_en), ("ko", args.extra_ko)) if p is not None}
    s = phonemize_manifests(args.manifest_dir, extra=extra, en_voice=args.en_voice, en_stress=args.en_stress)
    for split, f in s["files"].items():
        print(f"{split}: {f['rows']}줄 (빈 음소로 뺀 줄 {s['dropped_empty'][split]}) {f['path']}")
    c = s["characters"]
    print(f"글자 {c['count']}개(학습 줄 {c['from_training_rows']}개, 더한 글에서만 {c['only_from_extra']}개): {c['path']}")
    print(f"G2P 설정: 영어 {s['g2p']['en']['voice']}, 강세표 {'남김' if s['g2p']['en']['with_stress'] else '없음'} "
          f"-> {s['g2p']['path']}")
    if s["unknown_tokens"]:
        print(f"분류하지 못한 토큰: {s['unknown_tokens']}")
    print(f"걸린 시간 {s['seconds']}초")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
