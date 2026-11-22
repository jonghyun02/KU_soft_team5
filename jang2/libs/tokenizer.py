"""순수 파이썬 표준 라이브러리 기반 한국어 UTF-8 Byte-Fallback BPE 토크나이저 모듈."""

from __future__ import annotations

import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

TEXT_UNK_ID = 0
TEXT_BOS_ID = 1
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3

SPECIAL_TOKENS: Tuple[str, str, str, str] = ("<unk>", "<s>", "</s>", "<pad>")
SPACE_MARKER = "\u2581"

_PRETOKENIZE_RE = re.compile(
    r"\u2581?[가-힣ㄱ-ㅎㅏ-ㅣ]+"
    r"|\u2581?[A-Za-z]+"
    r"|\u2581?[0-9]+"
    r"|\u2581?[^\u2581가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+"
    r"|\u2581+"
)
_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class TokenizerProtocol(Protocol):
    """순수 파이썬 BPE 토크나이저와 SentencePiece가 공통으로 따르는 인터페이스 프로토콜입니다."""

    def EncodeAsIds(self, text: str) -> List[int]:
        """문자열을 서브워드 토큰 ID 리스트로 인코딩합니다."""
        ...

    def EncodeAsPieces(self, text: str) -> List[str]:
        """문자열을 서브워드 문자열 조각 리스트로 인코딩합니다."""
        ...

    def DecodeIds(self, ids: Sequence[int]) -> str:
        """토큰 ID 시퀀스를 원본 문자열로 디코딩합니다."""
        ...

    def GetPieceSize(self) -> int:
        """전체 어휘 사전 크기를 반환합니다."""
        ...

    def unk_id(self) -> int:
        """미등록(UNK) 특수 토큰 ID를 반환합니다."""
        ...

    def bos_id(self) -> int:
        """문장 시작(BOS) 특수 토큰 ID를 반환합니다."""
        ...

    def eos_id(self) -> int:
        """문장 종료(EOS) 특수 토큰 ID를 반환합니다."""
        ...

    def pad_id(self) -> int:
        """패딩(PAD) 특수 토큰 ID를 반환합니다."""
        ...


def normalize_korean_text(text: str) -> str:
    """입력 텍스트를 한글 완성형(NFC)으로 정규화하고 연속 공백을 단일 공백으로 정리합니다."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text))
    return " ".join(norm.strip().split())


def pretokenize_for_bpe(text: str) -> List[str]:
    """공백을 메타스페이스로 치환한 뒤 한글·영문·숫자·기호 경계를 보존하여 사전 토큰으로 분할합니다."""
    cleaned = normalize_korean_text(text)
    if not cleaned:
        return []
    escaped = SPACE_MARKER + cleaned.replace(" ", SPACE_MARKER)
    return _PRETOKENIZE_RE.findall(escaped)


class KoreanBPETokenizer:
    """외부 C++ 패키지 없이 동작하는 한국어 특화 UTF-8 Byte-Fallback BPE 서브워드 토크나이저입니다."""

    FORMAT_VERSION = "edarch_py_bpe_v1"

    def __init__(
        self,
        id_to_token: List[str],
        merges: List[Tuple[int, int, int]],
        byte_fallback: bool = True,
    ):
        """어휘 리스트와 BPE 병합 규칙으로부터 토크나이저 인스턴스를 초기화합니다."""
        if len(id_to_token) < 4 or tuple(id_to_token[:4]) != SPECIAL_TOKENS:
            raise ValueError(
                f"초기 특수 토큰 ID 0..3은 {SPECIAL_TOKENS}여야 합니다: {id_to_token[:4]}"
            )
        self.id_to_token: List[str] = list(id_to_token)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.merges: List[Tuple[int, int, int]] = [
            (int(a), int(b), int(m)) for a, b, m in merges
        ]
        self.merge_ranks: Dict[Tuple[int, int], Tuple[int, int]] = {
            (a, b): (rank, merged_id)
            for rank, (a, b, merged_id) in enumerate(self.merges)
        }
        self.byte_fallback = bool(byte_fallback)

        self._byte_to_id: Dict[int, int] = {}
        self._id_to_byte: Dict[int, int] = {}
        for idx, tok in enumerate(self.id_to_token):
            m = _BYTE_TOKEN_RE.match(tok)
            if m:
                b_val = int(m.group(1), 16)
                self._byte_to_id[b_val] = idx
                self._id_to_byte[idx] = b_val

        self._word_cache: Dict[str, Tuple[int, ...]] = {}
        self._max_cache_size = 65536

    @property
    def vocab_size(self) -> int:
        """등록된 전체 어휘 개수를 반환합니다."""
        return len(self.id_to_token)

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        vocab_size: int = 32000,
        min_pair_freq: int = 2,
        byte_fallback: bool = True,
    ) -> "KoreanBPETokenizer":
        """역인덱스와 지연 최대 힙을 사용하여 코퍼스로부터 BPE 어휘 사전과 병합 규칙을 학습합니다."""
        min_required = max(TEXT_EOS_ID, TEXT_PAD_ID) + 8
        if vocab_size < min_required:
            raise ValueError(f"vocab_size({vocab_size})는 최소 {min_required} 이상이어야 합니다.")

        pretoken_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()
        for sent in sentences:
            ptoks = pretokenize_for_bpe(sent)
            for pt in ptoks:
                pretoken_counts[pt] += 1
                for ch in pt:
                    char_counts[ch] += 1

        if not pretoken_counts:
            raise ValueError("토크나이저 학습을 위한 유효한 문장이 비어 있습니다.")

        use_bytes = bool(byte_fallback and vocab_size >= 300)
        id_to_token: List[str] = list(SPECIAL_TOKENS)
        token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(id_to_token)}

        byte_ids: Dict[int, int] = {}
        if use_bytes:
            for b in range(256):
                btok = f"<0x{b:02X}>"
                bid = len(id_to_token)
                id_to_token.append(btok)
                token_to_id[btok] = bid
                byte_ids[b] = bid

        remaining_slots = vocab_size - len(id_to_token)
        max_init_chars = (
            max(16, int(remaining_slots * 0.65))
            if use_bytes
            else max(8, int(remaining_slots * 0.70))
        )

        sorted_chars = sorted(char_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if SPACE_MARKER in char_counts and SPACE_MARKER not in token_to_id:
            token_to_id[SPACE_MARKER] = len(id_to_token)
            id_to_token.append(SPACE_MARKER)

        base_offset = 260 if use_bytes else 4
        for ch, _ in sorted_chars:
            if len(id_to_token) - base_offset >= max_init_chars:
                break
            if ch not in token_to_id:
                token_to_id[ch] = len(id_to_token)
                id_to_token.append(ch)

        words: List[List[int]] = []
        freqs: List[int] = []
        for ptok, freq in pretoken_counts.items():
            seq: List[int] = []
            for ch in ptok:
                tid = token_to_id.get(ch)
                if tid is not None:
                    seq.append(tid)
                elif use_bytes:
                    for b in ch.encode("utf-8"):
                        seq.append(byte_ids[b])
                else:
                    seq.append(TEXT_UNK_ID)
            if len(seq) >= 2:
                words.append(seq)
                freqs.append(freq)

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        for w_idx, seq in enumerate(words):
            w_freq = freqs[w_idx]
            pc = Counter(zip(seq, seq[1:]))
            for pair, cnt in pc.items():
                pair_counts[pair] += cnt * w_freq
                pair_to_words[pair].add(w_idx)

        eff_min_freq = max(1, int(min_pair_freq))
        heap: List[Tuple[int, int, int]] = [
            (-cnt, p[0], p[1])
            for p, cnt in pair_counts.items()
            if cnt >= eff_min_freq
        ]
        heapq.heapify(heap)

        merges: List[Tuple[int, int, int]] = []
        while len(id_to_token) < vocab_size and heap:
            neg_cnt, a, b = heapq.heappop(heap)
            pair = (a, b)
            actual = pair_counts.get(pair, 0)
            if actual < eff_min_freq:
                continue
            if -neg_cnt != actual:
                heapq.heappush(heap, (-actual, a, b))
                continue

            new_piece = id_to_token[a] + id_to_token[b]
            existing_id = token_to_id.get(new_piece)
            if existing_id is None:
                new_id = len(id_to_token)
                id_to_token.append(new_piece)
                token_to_id[new_piece] = new_id
            else:
                new_id = existing_id

            merges.append((a, b, new_id))

            affected = pair_to_words.pop(pair, set())
            pair_counts.pop(pair, None)
            dirty_pairs: Set[Tuple[int, int]] = set()

            for w_idx in affected:
                old_seq = words[w_idx]
                old_pc = Counter(zip(old_seq, old_seq[1:]))

                new_seq: List[int] = []
                i = 0
                n = len(old_seq)
                while i < n:
                    if i + 1 < n and old_seq[i] == a and old_seq[i + 1] == b:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1

                if new_seq == old_seq:
                    continue

                words[w_idx] = new_seq
                w_freq = freqs[w_idx]
                new_pc = Counter(zip(new_seq, new_seq[1:]))

                for p_old, c_old in old_pc.items():
                    if p_old == pair:
                        continue
                    pair_counts[p_old] -= c_old * w_freq
                    if p_old not in new_pc:
                        pair_to_words[p_old].discard(w_idx)
                    dirty_pairs.add(p_old)

                for p_new, c_new in new_pc.items():
                    pair_counts[p_new] += c_new * w_freq
                    pair_to_words[p_new].add(w_idx)
                    dirty_pairs.add(p_new)

            for dp in dirty_pairs:
                dp_cnt = pair_counts.get(dp, 0)
                if dp_cnt >= eff_min_freq:
                    heapq.heappush(heap, (-dp_cnt, dp[0], dp[1]))
                elif dp_cnt <= 0:
                    pair_counts.pop(dp, None)
                    pair_to_words.pop(dp, None)

        return cls(id_to_token=id_to_token, merges=merges, byte_fallback=use_bytes)

    def _encode_pretoken(self, ptok: str) -> Tuple[int, ...]:
        """단일 사전 토큰에 학습된 병합 우선순위를 반복 적용하고 결과를 캐싱합니다."""
        cached = self._word_cache.get(ptok)
        if cached is not None:
            return cached

        ids: List[int] = []
        for ch in ptok:
            tid = self.token_to_id.get(ch)
            if tid is not None:
                ids.append(tid)
            elif self.byte_fallback and self._byte_to_id:
                for b in ch.encode("utf-8"):
                    ids.append(self._byte_to_id[b])
            else:
                ids.append(TEXT_UNK_ID)

        while len(ids) >= 2:
            best_rank = 1_000_000_000
            best_pair: Optional[Tuple[int, int]] = None
            best_merged_id = -1

            for i in range(len(ids) - 1):
                cand = (ids[i], ids[i + 1])
                rank_info = self.merge_ranks.get(cand)
                if rank_info is not None and rank_info[0] < best_rank:
                    best_rank, best_merged_id = rank_info
                    best_pair = cand

            if best_pair is None:
                break

            pa, pb = best_pair
            merged_ids: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == pa and ids[i + 1] == pb:
                    merged_ids.append(best_merged_id)
                    i += 2
                else:
                    merged_ids.append(ids[i])
                    i += 1
            ids = merged_ids

        out_tuple = tuple(ids)
        if len(self._word_cache) < self._max_cache_size:
            self._word_cache[ptok] = out_tuple
        return out_tuple

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """문자열을 서브워드 토큰 ID 리스트로 변환합니다."""
        out: List[int] = []
        if add_bos:
            out.append(TEXT_BOS_ID)
        for ptok in pretokenize_for_bpe(text):
            out.extend(self._encode_pretoken(ptok))
        if add_eos:
            out.append(TEXT_EOS_ID)
        return out

    def encode_pieces(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[str]:
        """문자열을 서브워드 토큰 문자열 리스트로 변환합니다."""
        ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
        return [self.id_to_token[i] for i in ids]

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """토큰 ID 시퀀스를 UTF-8 바이트 폴백까지 조합하여 원본 한국어 문자열로 복원합니다."""
        pieces: List[str] = []
        byte_buf = bytearray()

        def _flush_bytes() -> None:
            """누적된 바이트 폴백 토큰 버퍼를 UTF-8 문자열로 디코딩해 조각 리스트에 추가합니다."""
            if byte_buf:
                pieces.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for raw_id in ids:
            tid = int(raw_id)
            if tid < 0 or tid >= len(self.id_to_token):
                continue
            if skip_special_tokens and tid in (
                TEXT_UNK_ID,
                TEXT_BOS_ID,
                TEXT_EOS_ID,
                TEXT_PAD_ID,
            ):
                _flush_bytes()
                continue

            b_val = self._id_to_byte.get(tid)
            if b_val is not None:
                byte_buf.append(b_val)
            else:
                _flush_bytes()
                pieces.append(self.id_to_token[tid])

        _flush_bytes()
        joined = "".join(pieces)
        if joined.startswith(SPACE_MARKER):
            joined = joined[len(SPACE_MARKER) :]
        return joined.replace(SPACE_MARKER, " ")

    def save(self, path: Path | str) -> Path:
        """학습된 어휘 사전과 병합 규칙을 UTF-8 JSON 파일로 저장합니다."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT_VERSION,
            "vocab_size": len(self.id_to_token),
            "byte_fallback": self.byte_fallback,
            "special_tokens": {
                "unk_id": TEXT_UNK_ID,
                "bos_id": TEXT_BOS_ID,
                "eos_id": TEXT_EOS_ID,
                "pad_id": TEXT_PAD_ID,
            },
            "tokens": self.id_to_token,
            "merges": [[a, b, m] for a, b, m in self.merges],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_path

    @classmethod
    def load(cls, path: Path | str) -> "KoreanBPETokenizer":
        """저장된 JSON 파일로부터 KoreanBPETokenizer 인스턴스를 복원합니다."""
        in_path = Path(path)
        raw = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != cls.FORMAT_VERSION:
            raise ValueError(f"유효한 KoreanBPETokenizer JSON 파일이 아닙니다: {in_path}")
        tokens = [str(t) for t in raw["tokens"]]
        merges = [(int(m[0]), int(m[1]), int(m[2])) for m in raw.get("merges", [])]
        byte_fb = bool(raw.get("byte_fallback", True))
        return cls(id_to_token=tokens, merges=merges, byte_fallback=byte_fb)

    def Load(self, path: str) -> bool:
        """SentencePiece 호환 방식으로 파일 경로에서 토크나이저 상태를 로드합니다."""
        loaded = self.load(path)
        self.__dict__.update(loaded.__dict__)
        return True

    def EncodeAsIds(self, text: str) -> List[int]:
        """SentencePiece 호환 방식으로 문자열을 토큰 ID 리스트로 인코딩합니다."""
        return self.encode(text, add_bos=False, add_eos=False)

    def EncodeAsPieces(self, text: str) -> List[str]:
        """SentencePiece 호환 방식으로 문자열을 서브워드 조각 리스트로 인코딩합니다."""
        return self.encode_pieces(text, add_bos=False, add_eos=False)

    def DecodeIds(self, ids: Sequence[int]) -> str:
        """SentencePiece 호환 방식으로 토큰 ID 시퀀스를 문자열로 디코딩합니다."""
        return self.decode(ids, skip_special_tokens=True)

    def GetPieceSize(self) -> int:
        """SentencePiece 호환 방식으로 전체 어휘 크기를 반환합니다."""
        return len(self.id_to_token)

    def IdToPiece(self, idx: int) -> str:
        """지정된 토큰 ID에 대응하는 서브워드 문자열을 반환합니다."""
        return self.id_to_token[idx]

    def PieceToId(self, piece: str) -> int:
        """지정된 서브워드 문자열에 대응하는 토큰 ID를 반환합니다."""
        return self.token_to_id.get(piece, TEXT_UNK_ID)

    def unk_id(self) -> int:
        """미등록(UNK) 특수 토큰 ID(0)를 반환합니다."""
        return TEXT_UNK_ID

    def bos_id(self) -> int:
        """문장 시작(BOS) 특수 토큰 ID(1)를 반환합니다."""
        return TEXT_BOS_ID

    def eos_id(self) -> int:
        """문장 종료(EOS) 특수 토큰 ID(2)를 반환합니다."""
        return TEXT_EOS_ID

    def pad_id(self) -> int:
        """패딩(PAD) 특수 토큰 ID(3)를 반환합니다."""
        return TEXT_PAD_ID


def is_pure_python_tokenizer_file(path: Path) -> bool:
    """주어진 파일이 순수 파이썬 JSON 토크나이저 포맷인지 판별합니다."""
    if path.suffix.lower() == ".json":
        return True
    if not path.exists():
        return False
    try:
        with path.open("rb") as f:
            head = f.read(16).lstrip()
        return head.startswith(b"{")
    except OSError:
        return False


def load_korean_tokenizer(path: Path | str) -> TokenizerProtocol:
    """순수 파이썬 JSON 토크나이저 또는 SentencePiece 바이너리 모델을 자동 감지해 로드합니다."""
    tok_path = Path(path)
    if not tok_path.exists():
        raise FileNotFoundError(f"토크나이저 파일을 찾을 수 없습니다: {tok_path}")

    if is_pure_python_tokenizer_file(tok_path):
        return KoreanBPETokenizer.load(tok_path)

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            f"{tok_path} 파일은 바이너리 SentencePiece 포맷이지만 sentencepiece가 설치되어 있지 않습니다. "
            "순수 파이썬 토크나이저(.json)를 사용하거나 sentencepiece를 설치하세요."
        ) from exc

    proc = spm.SentencePieceProcessor()
    proc.Load(str(tok_path))
    return proc


def run_tokenizer_self_test() -> None:
    """순수 파이썬 한국어 BPE 토크나이저의 학습·인코딩·바이트 폴백·무손실 복원 무결성을 검증합니다."""
    sample_corpus = [
        "안녕하세요, 국제 학술대회 동시통역 시스템 테스트입니다.",
        "인공지능 기반 실시간 음성 번역 모델은 낮은 지연시간과 높은 정확도를 목표로 합니다.",
        "영어 발화가 끝나기 전에 한국어 번역 토큰과 음성을 스트리밍으로 생성합니다.",
        "실시간 동시통역에서는 문맥을 파악하며 자연스러운 한국어 어순으로 번역해야 합니다.",
        "강화학습을 통해 지연시간과 번역 품질 사이의 균형을 스스로 최적화합니다.",
    ] * 8

    tok = KoreanBPETokenizer.train(sample_corpus, vocab_size=400, min_pair_freq=2, byte_fallback=True)
    test_sentences = [
        "안녕하세요, 실시간 동시통역 테스트입니다!",
        "미등록 한자(韓英)와 이모지(🚀) 및 숫자 2026도 무손실 복원됩니다.",
    ]

    print(f"[Self-Test] KoreanBPETokenizer 학습 완료 (vocab_size={tok.GetPieceSize()}, merges={len(tok.merges)})")
    assert tok.unk_id() == 0 and tok.bos_id() == 1 and tok.eos_id() == 2 and tok.pad_id() == 3

    for sent in test_sentences:
        ids = tok.EncodeAsIds(sent)
        pieces = tok.EncodeAsPieces(sent)
        decoded = tok.DecodeIds(ids)
        assert TEXT_UNK_ID not in ids
        assert all(tid not in (TEXT_EOS_ID, TEXT_PAD_ID) for tid in ids)
        assert decoded == normalize_korean_text(sent)
        print(f"  - 원문: {sent}")
        print(f"    서브워드({len(pieces)}개): {pieces[:12]}{' ...' if len(pieces) > 12 else ''}")
        print(f"    복원: {decoded}")

    print("[OK] 순수 파이썬 한국어 BPE 토크나이저 무결성 검증 통과!")


def extract_korean_sentences_from_path(input_path: Path) -> List[str]:
    """텍스트 파일(`.txt`) 또는 AI-Hub JSON 디렉터리에서 한국어 문장 목록을 추출합니다."""
    if not input_path.exists():
        raise FileNotFoundError(f"토크나이저 학습 입력 경로를 찾을 수 없습니다: {input_path}")

    if input_path.is_file():
        lines = input_path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    sentences: List[str] = []
    seen: Set[str] = set()
    for jf in sorted(input_path.rglob("*.json")):
        try:
            raw = json.loads(jf.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates: List[str] = []
        if isinstance(raw, dict):
            data_list = raw.get("data")
            if isinstance(data_list, list):
                for item in data_list:
                    if isinstance(item, dict):
                        txt = item.get("ko_mtpe") or item.get("ko") or item.get("ko_mt") or ""
                        if isinstance(txt, str) and txt.strip():
                            candidates.append(txt.strip())
            label_info = raw.get("라벨링") or raw.get("labeling")
            if isinstance(label_info, dict):
                txt = label_info.get("한국어") or label_info.get("ko") or ""
                if isinstance(txt, str) and txt.strip():
                    candidates.append(txt.strip())
        for c in candidates:
            if c not in seen:
                seen.add(c)
                sentences.append(c)
    return sentences


def train_bpe_tokenizer_from_cli(
    input_path: Path,
    output_path: Path,
    vocab_size: int = 32000,
    min_freq: int = 2,
    byte_fallback: bool = True,
) -> Path:
    """입력 코퍼스 경로에서 한국어 문장을 읽어 순수 파이썬 BPE 토크나이저(`.json`)를 학습·저장합니다."""
    sentences = extract_korean_sentences_from_path(input_path)
    if not sentences:
        raise ValueError(f"입력 경로에서 유효한 한국어 문장을 찾지 못했습니다: {input_path}")
    tok = KoreanBPETokenizer.train(
        sentences=sentences,
        vocab_size=vocab_size,
        min_pair_freq=min_freq,
        byte_fallback=byte_fallback,
    )
    saved_path = tok.save(output_path)
    print(
        f"[Done] 순수 파이썬 BPE 토크나이저 저장 완료: {saved_path} "
        f"(문장 수={len(sentences)}, vocab_size={tok.GetPieceSize()}, merges={len(tok.merges)})"
    )
    return saved_path
