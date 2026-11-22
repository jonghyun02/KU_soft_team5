"""모음 거리표와 DTW 국소 비용 행렬.

논문(식 8, 9)은 Your-TTS 텍스트 인코더의 모음 임베딩 거리를 썼다. 여기서는 텍스트만 있으므로
eSpeak IPA 기호를 PanPhon 조음 자질 벡터(24차원, +1/0/-1)로 바꿔 유클리드 거리를 재고,
비대각 최댓값으로 나눠 [0, 1] 로 맞춘다.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .config import DATA_DIR
from .phonetics import EN_VOWELS, KO_VOWEL_JAMO, KO_VOWELS

if TYPE_CHECKING:
    import panphon

METHOD = "panphon FeatureTable 24차원 자질 벡터의 유클리드 거리 / 비대각 최댓값"
TABLE_JSON = DATA_DIR / "vowel_table.json"
TABLE_CSV = DATA_DIR / "vowel_table_ko_en.csv"

# 표의 기호 순서(혀 앞 -> 뒤, 높음 -> 낮음). 여기 없는 기호는 뒤에 사전순으로 붙는다.
VOWEL_ORDER = ("i", "ɪ", "e", "ɛ", "æ", "ɨ", "ɯ", "ə", "ɜ", "ʌ", "a", "ɑ", "u", "ʊ", "o", "ɔ")


@dataclass(eq=False)
class VowelTable:
    """모음 기호 목록과 거리 행렬(정규화 [0,1], 대칭, 대각 0)."""
    symbols: list[str]
    matrix: np.ndarray
    method: str = METHOD
    vectors: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {s: i for i, s in enumerate(self.symbols)}

    def distance(self, a: str, b: str) -> float:
        return float(self.matrix[self.index[a], self.index[b]])


def ordered_symbols(symbols: Iterable[str]) -> list[str]:
    """기호를 VOWEL_ORDER 순서로 늘어놓는다(겹침 제거)."""
    uniq = set(symbols)
    head = [s for s in VOWEL_ORDER if s in uniq]
    return head + sorted(uniq - set(head))


def table_symbols() -> list[str]:
    """거리표에 들어갈 기호: 영어·한국어 모음의 합집합."""
    return ordered_symbols(EN_VOWELS | KO_VOWELS)


@lru_cache(maxsize=1)
def feature_table() -> panphon.FeatureTable:
    """PanPhon FeatureTable(만드는 데 0.4초쯤 걸려서 한 번만 만든다)."""
    import panphon

    try:
        return panphon.FeatureTable()
    except UnicodeDecodeError as e:
        # panphon 0.22.2 는 자료 CSV 를 인코딩 지정 없이 열어서 한국어 Windows(cp949)에서 깨진다.
        raise RuntimeError("PanPhon 자료 파일을 시스템 기본 인코딩으로 읽다 실패했습니다. "
                           "PYTHONUTF8=1 을 켜고 다시 실행하세요(run.cmd 는 켜 둡니다).") from e


def panphon_vector(symbol: str) -> list[int]:
    """기호 하나의 PanPhon 자질 벡터. 정확히 한 분절이 아니면 예외."""
    segs = feature_table().word_to_vector_list(symbol, numeric=True)
    if len(segs) != 1:
        raise ValueError(f"PanPhon 이 '{symbol}' 를 분절 {len(segs)}개로 읽습니다. 한 분절이어야 합니다.")
    return [int(v) for v in segs[0]]


def build_vowel_table(symbols: Sequence[str]) -> VowelTable:
    """기호마다 PanPhon 자질 벡터를 얻어 거리표를 만든다."""
    syms = list(dict.fromkeys(symbols))
    if len(syms) < 2:
        raise ValueError("모음표에는 기호가 두 개 이상 있어야 합니다.")
    vectors = {s: panphon_vector(s) for s in syms}
    x = np.array([vectors[s] for s in syms], dtype=np.float64)
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
    off = dist[~np.eye(len(syms), dtype=bool)]
    top = float(off.max())
    if top <= 0:
        raise ValueError("모든 기호의 자질 벡터가 같아서 거리를 정규화할 수 없습니다.")
    matrix = dist / top
    np.fill_diagonal(matrix, 0.0)
    return VowelTable(symbols=syms, matrix=matrix, method=METHOD, vectors=vectors)


def cost_matrix(src_labels: Sequence[str | None], tgt_labels: Sequence[str | None],
                table: VowelTable, null_cost: float) -> np.ndarray:
    """DTW 국소 비용 행렬 [len(src), len(tgt)].

    라벨은 모음 기호 또는 None(모음이 아닌 칸). d(None, None) = 0, d(None, 모음) = null_cost,
    모음끼리는 거리표 값이다. null 행·열을 덧붙인 확장 행렬을 넘파이 인덱싱으로 한 번에 뽑는다.
    """
    k = len(table.symbols)
    ext = np.zeros((k + 1, k + 1), dtype=np.float64)
    ext[:k, :k] = table.matrix
    ext[k, :k] = null_cost
    ext[:k, k] = null_cost

    def to_index(labels: Sequence[str | None]) -> np.ndarray:
        try:
            return np.fromiter((k if lab is None else table.index[lab] for lab in labels),
                               dtype=np.intp, count=len(labels))
        except KeyError as e:
            raise KeyError(f"모음표에 없는 기호입니다: {e.args[0]!r}") from None

    return ext[np.ix_(to_index(src_labels), to_index(tgt_labels))]


def _ko_en_rows(table: VowelTable) -> tuple[list[str], list[str]]:
    ko = [s for s in table.symbols if s in KO_VOWELS]
    en = [s for s in table.symbols if s in EN_VOWELS]
    return ko, en


def nearest_english(table: VowelTable) -> dict[str, list[tuple[str, float]]]:
    """한국어 모음마다 영어 모음을 가까운 순으로 늘어놓는다."""
    ko, en = _ko_en_rows(table)
    return {k: sorted(((e, table.distance(k, e)) for e in en), key=lambda p: (p[1], en.index(p[0])))
            for k in ko}


def _write_atomic(path: Path, text: str, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding, newline="")
    os.replace(tmp, path)


def save_table(table: VowelTable, json_path: Path = TABLE_JSON, csv_path: Path = TABLE_CSV) -> None:
    """캐시 JSON 과 한국어(행) x 영어(열) CSV 를 쓴다."""
    import panphon

    def one_line(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False)

    head = {
        "method": table.method,
        "panphon_version": getattr(panphon, "__version__", None) or _dist_version("panphon"),
        "features": list(feature_table().names),
        "ko_vowels": [s for s in table.symbols if s in KO_VOWELS],
        "en_vowels": [s for s in table.symbols if s in EN_VOWELS],
        "symbols": table.symbols,
    }
    # 행렬과 자질 벡터는 한 행에 한 줄로 적어 눈으로 읽기 쉽게 한다.
    parts = [f" {one_line(k)}: {one_line(v)}" for k, v in head.items()]
    rows = ",\n".join(f"  {one_line([round(float(v), 6) for v in row])}" for row in table.matrix)
    parts.append(f' "matrix": [\n{rows}\n ]')
    vecs = ",\n".join(f"  {one_line(s)}: {one_line(table.vectors[s])}" for s in table.symbols if s in table.vectors)
    parts.append(f' "vectors": {{\n{vecs}\n }}')
    text = "{\n" + ",\n".join(parts) + "\n}\n"
    json.loads(text)                                   # 쓰기 전에 형식 확인
    _write_atomic(json_path, text, "utf-8")

    ko, en = _ko_en_rows(table)
    lines: list[list[str]] = [["ko", "jamo", *en]]
    for k in ko:
        lines.append([k, KO_VOWEL_JAMO.get(k, ""), *(f"{table.distance(k, e):.4f}" for e in en)])
    buf = StringIO()
    csv.writer(buf, lineterminator="\n").writerows(lines)
    _write_atomic(csv_path, buf.getvalue(), "utf-8-sig")


def _dist_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_table(json_path: Path = TABLE_JSON) -> VowelTable:
    """캐시 JSON 을 읽는다. 행렬 꼴이 기호 수와 안 맞으면 예외."""
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    symbols = list(doc["symbols"])
    matrix = np.array(doc["matrix"], dtype=np.float64)
    if matrix.shape != (len(symbols), len(symbols)):
        raise ValueError(f"모음표 행렬 꼴 {matrix.shape} 가 기호 {len(symbols)}개와 맞지 않습니다: {json_path}")
    return VowelTable(symbols=symbols, matrix=matrix, method=doc.get("method", METHOD),
                      vectors=doc.get("vectors", {}))


_CACHE: dict[Path, VowelTable] = {}


def load_or_build(json_path: Path = TABLE_JSON, rebuild: bool = False) -> VowelTable:
    """캐시가 있고 지금 쓰는 기호·방법과 같으면 읽고, 아니면 새로 만들어 저장한다."""
    key = Path(json_path).resolve()
    if not rebuild and key in _CACHE:
        return _CACHE[key]
    want = table_symbols()
    table = None
    if not rebuild and key.exists():
        try:
            cached = load_table(key)
        except (OSError, ValueError, KeyError):
            cached = None
        if cached is not None and cached.symbols == want and cached.method == METHOD:
            table = cached
    if table is None:
        table = build_vowel_table(want)
        save_table(table, key, key.with_name(TABLE_CSV.name))
    _CACHE[key] = table
    return table
