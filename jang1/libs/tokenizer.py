"""arXiv:2309.11093 제어 토큰(<SYL_N>, <SEP>, 섹션·반복 지시자) 내장 한-영 SentencePiece 토크나이저 모듈."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from libs.phonetics import count_line_syllables, normalize_lyric_text

TEXT_UNK_ID = 0
SPM_BOS_ID = 1
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3

SEP_TOKEN = "<SEP>"
KEEP_CS_TOKEN = "<KEEP_CS>"
MAX_SYLLABLE_TOKEN = 64
MAX_REPEAT_TOKEN = 16
SPACE_MARKER = "\u2581"

SECTION_TOKENS: Tuple[str, ...] = (
    "<SEC_VERSE>",
    "<SEC_PRECHORUS>",
    "<SEC_CHORUS>",
    "<SEC_HOOK>",
    "<SEC_BRIDGE>",
    "<SEC_INTRO>",
    "<SEC_OUTRO>",
    "<SEC_OTHER>",
)
SYLLABLE_TOKENS: Tuple[str, ...] = tuple(
    f"<SYL{i}>" for i in range(1, MAX_SYLLABLE_TOKEN + 1)
)
REPEAT_TOKENS: Tuple[str, ...] = tuple(
    f"<REP{i}>" for i in range(MAX_REPEAT_TOKEN)
)
CONTROL_USER_SYMBOLS: Tuple[str, ...] = (
    (SEP_TOKEN, KEEP_CS_TOKEN) + SECTION_TOKENS + SYLLABLE_TOKENS + REPEAT_TOKENS
)

_CONTROL_SPLIT_RE = re.compile(
    r"\s*(?:<SEP>|<SYL\d+>|<REP\d+>|<SEC_[A-Z]+>|<KEEP_CS>)\s*"
)
_LINE_BOUNDARY_RE = re.compile(r"\s*<SEP>\s*|\s*(?=<SYL\d+>)")
_MAX_VOCAB_HINT_RE = re.compile(r"<=\s*(\d+)")
_MIN_VOCAB_HINT_RE = re.compile(r"smaller than required_chars\.\s*\d+\s*vs\s*(\d+)")
_FALLBACK_MAGIC = b"LYRICSPM1\x00"


class TokenizerProtocol(Protocol):
    """노래 번역 파이프라인에서 사용하는 SentencePiece 호환 토크나이저 인터페이스 프로토콜입니다."""

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

    def IdToPiece(self, idx: int) -> str:
        """토큰 ID에 대응하는 서브워드 문자열을 반환합니다."""
        ...

    def PieceToId(self, piece: str) -> int:
        """서브워드 문자열에 대응하는 토큰 ID를 반환합니다."""
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


class FallbackLyricSubwordProcessor:
    """sentencepiece C++ 확장이 없는 환경에서도 동일한 바이너리 `.model` 인터페이스를 제공하는 내장 토크나이저입니다."""

    def __init__(self, pieces: Sequence[str]):
        """서브워드 조각 목록으로부터 정방향·역방향 어휘 사전을 구축합니다."""
        self._id_to_piece = list(pieces)
        self._piece_to_id: Dict[str, int] = {p: i for i, p in enumerate(self._id_to_piece)}
        self._ctrl_set = set(CONTROL_USER_SYMBOLS)

    @classmethod
    def train_from_sentences(
        cls,
        sentences: Sequence[str],
        vocab_size: int,
    ) -> "FallbackLyricSubwordProcessor":
        """코퍼스 내 단어·문자 빈도와 바이트 폴백을 결합해 지정된 크기의 서브워드 어휘 사전을 구성합니다."""
        pieces: List[str] = ["<unk>", "<s>", "</s>", "<pad>"]
        pieces.extend(CONTROL_USER_SYMBOLS)

        seen = set(pieces)
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()
        ctrl_set = set(CONTROL_USER_SYMBOLS)

        for sent in sentences:
            for tok in sent.split():
                if tok in ctrl_set:
                    continue
                marked = f"{SPACE_MARKER}{tok}"
                word_counts[marked] += 1
                for ch in marked:
                    char_counts[ch] += 1

        for ch, _ in char_counts.most_common():
            if len(pieces) >= vocab_size:
                break
            if ch not in seen:
                seen.add(ch)
                pieces.append(ch)

        for w, _ in word_counts.most_common():
            if len(pieces) >= vocab_size:
                break
            if w not in seen:
                seen.add(w)
                pieces.append(w)

        for b in range(256):
            if len(pieces) >= vocab_size:
                break
            bp = f"<0x{b:02X}>"
            if bp not in seen:
                seen.add(bp)
                pieces.append(bp)

        return cls(pieces)

    def save(self, path: Path) -> None:
        """어휘 사전을 바이너리 `.model` 파일로 직렬화하여 저장합니다."""
        payload = _FALLBACK_MAGIC + pickle.dumps(self._id_to_piece, protocol=4)
        path.write_bytes(payload)

    @classmethod
    def load(cls, path: Path) -> "FallbackLyricSubwordProcessor":
        """직렬화된 바이너리 `.model` 파일에서 어휘 사전을 복원합니다."""
        raw = path.read_bytes()
        pieces = pickle.loads(raw[len(_FALLBACK_MAGIC) :])
        return cls(pieces)

    def EncodeAsPieces(self, text: str) -> List[str]:
        """문자열을 제어 토큰, 어휘 사전 서브워드 및 UTF-8 바이트 폴백 조각으로 분해합니다."""
        norm = normalize_lyric_text(text)
        if not norm:
            return []
        out: List[str] = []
        for word in norm.split():
            if word in self._ctrl_set:
                out.append(word)
                continue
            cand = f"{SPACE_MARKER}{word}"
            if cand in self._piece_to_id:
                out.append(cand)
                continue
            for ch in cand:
                if ch in self._piece_to_id:
                    out.append(ch)
                else:
                    for b in ch.encode("utf-8"):
                        out.append(f"<0x{b:02X}>")
        return out

    def EncodeAsIds(self, text: str) -> List[int]:
        """문자열을 서브워드 토큰 ID 리스트로 인코딩합니다."""
        return [self._piece_to_id.get(p, TEXT_UNK_ID) for p in self.EncodeAsPieces(text)]

    def DecodeIds(self, ids: Sequence[int]) -> str:
        """토큰 ID 리스트를 UTF-8 바이트 병합을 거쳐 원본 문자열로 복원합니다."""
        pieces: List[str] = []
        byte_buf = bytearray()

        def _flush_bytes() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="ignore"))
                byte_buf.clear()

        for tid in ids:
            idx = int(tid)
            if idx in (TEXT_UNK_ID, SPM_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID):
                continue
            if not (0 <= idx < len(self._id_to_piece)):
                continue
            p = self._id_to_piece[idx]
            if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
                try:
                    byte_buf.append(int(p[3:5], 16))
                    continue
                except ValueError:
                    pass
            _flush_bytes()
            if p in self._ctrl_set:
                pieces.append(f" {p} ")
            else:
                pieces.append(p)
        _flush_bytes()
        joined = "".join(pieces).replace(SPACE_MARKER, " ")
        return normalize_lyric_text(joined)

    def GetPieceSize(self) -> int:
        """전체 어휘 사전 크기를 반환합니다."""
        return len(self._id_to_piece)

    def IdToPiece(self, idx: int) -> str:
        """토큰 ID에 대응하는 서브워드 문자열을 반환합니다."""
        return self._id_to_piece[idx]

    def PieceToId(self, piece: str) -> int:
        """서브워드 문자열에 대응하는 토큰 ID를 반환합니다."""
        return self._piece_to_id.get(piece, TEXT_UNK_ID)

    def unk_id(self) -> int:
        """미등록(UNK) 특수 토큰 ID를 반환합니다."""
        return TEXT_UNK_ID

    def bos_id(self) -> int:
        """문장 시작(BOS) 특수 토큰 ID를 반환합니다."""
        return SPM_BOS_ID

    def eos_id(self) -> int:
        """문장 종료(EOS) 특수 토큰 ID를 반환합니다."""
        return TEXT_EOS_ID

    def pad_id(self) -> int:
        """패딩(PAD) 특수 토큰 ID를 반환합니다."""
        return TEXT_PAD_ID


def format_syllable_token(syllable_count: int) -> str:
    """목표 음절 수 정수를 `<SYL1>` ~ `<SYL64>` 제어 토큰 문자열로 변환합니다."""
    clamped = max(1, min(int(syllable_count), MAX_SYLLABLE_TOKEN))
    return f"<SYL{clamped}>"


def format_repeat_token(repeat_group_id: int) -> str:
    """섹션 내 반복 행 그룹 ID를 `<REP0>` ~ `<REP15>` 제어 토큰 문자열로 변환합니다."""
    clamped = max(0, min(int(repeat_group_id), MAX_REPEAT_TOKEN - 1))
    return f"<REP{clamped}>"


def format_section_token(section_type_id: int) -> str:
    """섹션 타입 ID(0..7)를 `<SEC_VERSE>`, `<SEC_CHORUS>` 등의 제어 토큰으로 변환합니다."""
    idx = max(0, min(int(section_type_id), len(SECTION_TOKENS) - 1))
    return SECTION_TOKENS[idx]


def render_section_with_controls(
    lines: Sequence[str],
    target_syllables: Optional[Sequence[int]] = None,
    repeat_group_ids: Optional[Sequence[int]] = None,
    section_type_id: Optional[int] = None,
    include_syllable_tokens: bool = True,
) -> str:
    """섹션 내 가사 행들에 목표 음절 수(`<SYL_N>`), 반복 지시자(`<REP_K>`), 행 구분자(`<SEP>`)를 결합합니다."""
    cleaned_lines = [normalize_lyric_text(ln) for ln in lines if normalize_lyric_text(ln)]
    if not cleaned_lines:
        return ""

    n = len(cleaned_lines)
    if target_syllables is not None and len(target_syllables) >= n:
        syl_list = [max(1, int(s)) for s in target_syllables[:n]]
    else:
        syl_list = [max(1, count_line_syllables(ln)) for ln in cleaned_lines]

    rendered_parts: List[str] = []
    for idx, ln in enumerate(cleaned_lines):
        prefix_tokens: List[str] = []
        if include_syllable_tokens:
            prefix_tokens.append(format_syllable_token(syl_list[idx]))
        if repeat_group_ids is not None and idx < len(repeat_group_ids):
            prefix_tokens.append(format_repeat_token(repeat_group_ids[idx]))
        if prefix_tokens:
            rendered_parts.append(f"{' '.join(prefix_tokens)} {ln}")
        else:
            rendered_parts.append(ln)

    body = f" {SEP_TOKEN} ".join(rendered_parts)
    if section_type_id is not None:
        return f"{format_section_token(section_type_id)} {body}"
    return body


def split_rendered_section_lines(
    rendered_text: str,
    expected_num_lines: Optional[int] = None,
) -> List[str]:
    """모델이 생성한 섹션 문자열에서 제어 토큰을 분리하고 원문 행 수(`expected_num_lines`)에 맞춰 행 리스트로 복원합니다."""
    cleaned = rendered_text.replace("</s>", "").replace("<pad>", "").replace("<s>", "").strip()
    cleaned = re.sub(r"<SEC_[A-Z]+>\s*", "", cleaned)
    cleaned = re.sub(r"<REP\d+>\s*", "", cleaned)
    cleaned = re.sub(r"<KEEP_CS>\s*", "", cleaned)

    raw_chunks = [c.strip() for c in _LINE_BOUNDARY_RE.split(cleaned) if c.strip()]
    lines: List[str] = []
    for chunk in raw_chunks:
        stripped = _CONTROL_SPLIT_RE.sub(" ", chunk).strip()
        stripped = normalize_lyric_text(stripped)
        if stripped:
            lines.append(stripped)

    if not lines and cleaned:
        fallback = normalize_lyric_text(_CONTROL_SPLIT_RE.sub(" ", cleaned))
        if fallback:
            lines = [fallback]

    if expected_num_lines is None or expected_num_lines <= 0:
        return lines

    if len(lines) == expected_num_lines:
        return lines
    if len(lines) > expected_num_lines:
        return lines[: expected_num_lines - 1] + [" ".join(lines[expected_num_lines - 1 :])]
    if len(lines) == 1 and expected_num_lines > 1:
        words = lines[0].split()
        if len(words) >= expected_num_lines:
            split_out: List[str] = []
            for i in range(expected_num_lines):
                s_i = (i * len(words)) // expected_num_lines
                e_i = ((i + 1) * len(words)) // expected_num_lines
                split_out.append(" ".join(words[s_i:e_i]))
            return split_out
    return lines + [""] * (expected_num_lines - len(lines))


def resolve_spm_model_path(path: Path | str) -> Path:
    """출력 경로에 `.model` 확장자가 없으면 자동으로 부여하여 정규화된 경로를 반환합니다."""
    out_path = Path(path)
    if out_path.suffix.lower() == ".json":
        raise ValueError(
            f"JSON 토크나이저 포맷은 지원되지 않습니다: {out_path}. "
            "SentencePiece 모델 파일 경로(예: lyric_spm.model)를 지정하세요."
        )
    if out_path.suffix.lower() != ".model":
        return Path(f"{out_path}.model")
    return out_path


def load_lyric_tokenizer(path: Path | str) -> TokenizerProtocol:
    """SentencePiece 또는 내장 서브워드 바이너리 `.model` 파일을 로드하고 특수 토큰 규격을 검증합니다."""
    tok_path = Path(path)
    if not tok_path.exists():
        raise FileNotFoundError(f"SentencePiece 토크나이저 파일을 찾을 수 없습니다: {tok_path}")
    if tok_path.suffix.lower() == ".json":
        raise ValueError(f"바이너리 SentencePiece(.model) 파일을 지정하세요: {tok_path}")

    head = tok_path.read_bytes()[: len(_FALLBACK_MAGIC)]
    if head == _FALLBACK_MAGIC:
        return FallbackLyricSubwordProcessor.load(tok_path)

    if spm is None:
        raise RuntimeError(f"sentencepiece 패키지가 설치되어 있지 않아 {tok_path}를 로드할 수 없습니다.")
    proc = spm.SentencePieceProcessor()
    proc.Load(str(tok_path))
    return proc


def train_lyric_sentencepiece(
    sentences: Sequence[str],
    output_model_path: Path | str,
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
    byte_fallback: bool = True,
) -> Path:
    """한-영 병렬 가사 코퍼스로부터 arXiv:2309.11093 제어 토큰이 등록된 SentencePiece `.model`을 학습합니다."""
    num_ctrl = len(CONTROL_USER_SYMBOLS)
    min_required = max(TEXT_EOS_ID, TEXT_PAD_ID) + num_ctrl + 16
    if vocab_size < min_required:
        vocab_size = min_required
    if model_type not in ("unigram", "bpe", "char", "word"):
        raise ValueError(f"지원하지 않는 SentencePiece model_type입니다: {model_type}")

    cleaned_lines: List[str] = []
    for line in sentences:
        norm_line = normalize_lyric_text(line)
        if norm_line:
            cleaned_lines.append(norm_line)
    if not cleaned_lines:
        raise ValueError("SentencePiece 학습을 위한 유효한 가사 문장이 비어 있습니다.")

    final_model_path = resolve_spm_model_path(output_model_path).resolve()
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    if spm is None:
        fb_proc = FallbackLyricSubwordProcessor.train_from_sentences(
            sentences=cleaned_lines,
            vocab_size=vocab_size,
        )
        fb_proc.save(final_model_path)
        return final_model_path

    model_prefix = str(final_model_path.with_suffix(""))
    use_byte_fb = bool(byte_fallback and vocab_size >= 400)
    unique_chars = len({ch for s in cleaned_lines for ch in s if not ch.isspace()})
    base_pieces = (260 if use_byte_fb else 4) + num_ctrl
    min_needed_vocab = base_pieces + unique_chars + 8

    if vocab_size < min_needed_vocab and use_byte_fb:
        use_byte_fb = False
        base_pieces = 4 + num_ctrl
        min_needed_vocab = base_pieces + unique_chars + 8

    corpus_upper = base_pieces + max(64, unique_chars * 4, len(cleaned_lines) * 4)
    cur_vocab = min(vocab_size, max(min_needed_vocab, corpus_upper))
    cur_vocab = max(min_required, cur_vocab)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as tmp_f:
        tmp_corpus_path = Path(tmp_f.name)
        for line in cleaned_lines:
            tmp_f.write(line + "\n")

    try:
        for _ in range(12):
            try:
                spm.SentencePieceTrainer.Train(
                    input=str(tmp_corpus_path),
                    model_prefix=model_prefix,
                    vocab_size=cur_vocab,
                    character_coverage=character_coverage,
                    model_type=model_type,
                    byte_fallback=use_byte_fb,
                    unk_id=TEXT_UNK_ID,
                    bos_id=SPM_BOS_ID,
                    eos_id=TEXT_EOS_ID,
                    pad_id=TEXT_PAD_ID,
                    user_defined_symbols=list(CONTROL_USER_SYMBOLS),
                    hard_vocab_limit=False,
                    minloglevel=2,
                )
                break
            except RuntimeError as exc:
                msg = str(exc)
                if "Vocabulary size too high" in msg:
                    m_max = _MAX_VOCAB_HINT_RE.search(msg)
                    if m_max is not None:
                        hinted = int(m_max.group(1))
                        if min_required <= hinted < cur_vocab:
                            cur_vocab = hinted
                            continue
                    if cur_vocab > min_required:
                        cur_vocab = max(min_required, int(cur_vocab * 0.8))
                        continue
                if "smaller than required_chars" in msg:
                    m_min = _MIN_VOCAB_HINT_RE.search(msg)
                    needed = int(m_min.group(1)) if m_min is not None else (cur_vocab + 64)
                    if needed > cur_vocab:
                        cur_vocab = needed
                        continue
                    if use_byte_fb:
                        use_byte_fb = False
                        cur_vocab = max(min_required, unique_chars + num_ctrl + 16)
                        continue
                raise
    finally:
        try:
            tmp_corpus_path.unlink(missing_ok=True)
        except OSError:
            pass

    proc = load_lyric_tokenizer(final_model_path)
    if (
        proc.unk_id() != TEXT_UNK_ID
        or proc.bos_id() != SPM_BOS_ID
        or proc.eos_id() != TEXT_EOS_ID
        or proc.pad_id() != TEXT_PAD_ID
    ):
        raise RuntimeError(
            f"학습된 SentencePiece 특수 토큰 ID가 규격(0,1,2,3)과 다릅니다: "
            f"unk={proc.unk_id()}, bos={proc.bos_id()}, eos={proc.eos_id()}, pad={proc.pad_id()}"
        )
    return final_model_path


def extract_lyric_sentences_from_path(input_path: Path) -> List[str]:
    """텍스트(`.txt`) 또는 가사 JSON/JSONL 파일·디렉터리에서 한-영 가사 문장 코퍼스를 추출합니다."""
    if not input_path.exists():
        raise FileNotFoundError(f"토크나이저 학습 입력 경로를 찾을 수 없습니다: {input_path}")

    raw_sentences: List[str] = []

    def _collect_from_obj(obj: Any) -> None:
        if isinstance(obj, str):
            norm = normalize_lyric_text(obj)
            if norm:
                raw_sentences.append(norm)
        elif isinstance(obj, list):
            for elem in obj:
                _collect_from_obj(elem)
        elif isinstance(obj, dict):
            for key in ("src", "tgt", "ko", "en", "kr", "lines", "sections", "lyrics", "text"):
                if key in obj:
                    _collect_from_obj(obj[key])
            for val in obj.values():
                if isinstance(val, (list, dict)):
                    _collect_from_obj(val)

    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    for fp in files:
        if not fp.is_file():
            continue
        sfx = fp.suffix.lower()
        try:
            if sfx == ".json":
                _collect_from_obj(json.loads(fp.read_text(encoding="utf-8-sig")))
            elif sfx == ".jsonl":
                for ln in fp.read_text(encoding="utf-8-sig").splitlines():
                    if ln.strip():
                        _collect_from_obj(json.loads(ln))
            elif sfx in (".txt", ".tsv", ".csv"):
                for ln in fp.read_text(encoding="utf-8-sig").splitlines():
                    for part in ln.split("\t"):
                        norm = normalize_lyric_text(part)
                        if norm:
                            raw_sentences.append(norm)
        except (OSError, json.JSONDecodeError):
            continue

    seen: Set[str] = set()
    unique_sentences: List[str] = []
    for sent in raw_sentences:
        if sent not in seen:
            seen.add(sent)
            unique_sentences.append(sent)
    return unique_sentences


def train_spm_tokenizer_from_cli(
    input_path: Path,
    output_path: Path,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    byte_fallback: bool = True,
) -> Path:
    """CLI 입력 경로에서 한-영 가사 문장을 추출하여 SentencePiece(`.model`) 토크나이저를 학습·저장합니다."""
    sentences = extract_lyric_sentences_from_path(input_path=input_path)
    if not sentences:
        raise ValueError(f"입력 경로에서 유효한 가사 문장을 찾지 못했습니다: {input_path}")
    saved_path = train_lyric_sentencepiece(
        sentences=sentences,
        output_model_path=output_path,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        byte_fallback=byte_fallback,
    )
    proc = load_lyric_tokenizer(saved_path)
    print(
        f"[Done] 노래 가사 SentencePiece 토크나이저 저장 완료: {saved_path} "
        f"(문장 수={len(sentences)}, vocab_size={proc.GetPieceSize()}, model_type={model_type})"
    )
    return saved_path


def run_tokenizer_self_test() -> None:
    """한-영 통합 가사 SentencePiece 토크나이저와 arXiv:2309.11093 제어 토큰 렌더링·복원을 검증합니다."""
    sample_corpus = [
        "빛나는 별빛 아래 너와 나 손을 잡고 걸어가",
        "Under the shining starlight you and I walk hand in hand",
        "Oh baby 멈추지 마 tonight 우리의 노래를 불러줘",
        "Oh baby don't stop tonight sing our melody out loud",
        "다시 시작해 새로운 아침이 밝아와",
        "Starting all over again as the bright morning comes",
    ] * 8

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = train_lyric_sentencepiece(
            sentences=sample_corpus,
            output_model_path=Path(tmp_dir) / "self_test_lyric.model",
            vocab_size=600,
            model_type="unigram",
            byte_fallback=True,
        )
        tok = load_lyric_tokenizer(model_path)

        sec_lines = [
            "Under the shining starlight you and I",
            "Oh baby don't stop tonight",
            "Under the shining starlight you and I",
        ]
        rendered = render_section_with_controls(
            lines=sec_lines,
            target_syllables=[10, 7, 10],
            repeat_group_ids=[0, 1, 0],
            section_type_id=2,
        )
        ids = list(tok.EncodeAsIds(rendered))
        pieces = list(tok.EncodeAsPieces(rendered))
        decoded = tok.DecodeIds(ids)
        recovered_lines = split_rendered_section_lines(decoded, expected_num_lines=3)

        assert tok.unk_id() == TEXT_UNK_ID
        assert tok.bos_id() == SPM_BOS_ID
        assert tok.eos_id() == TEXT_EOS_ID
        assert tok.pad_id() == TEXT_PAD_ID
        assert "<SYL10>" in pieces and "<SEP>" in pieces and "<SEC_CHORUS>" in pieces
        assert recovered_lines == sec_lines, f"섹션 행 복원 불일치: {recovered_lines}"

        print(
            f"[Self-Test] Lyric SentencePiece 학습·로드 완료 "
            f"(vocab_size={tok.GetPieceSize()}, 제어토큰={len(CONTROL_USER_SYMBOLS)}개)"
        )
        print(f"  - 렌더링된 섹션: {rendered}")
        print(f"  - 복원된 가사 행({len(recovered_lines)}행): {recovered_lines}")

    print("[OK] 한-영 노래 가사 SentencePiece 토크나이저 무결성 검증 통과!")
