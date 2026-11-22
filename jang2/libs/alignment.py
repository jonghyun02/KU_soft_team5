"""타깃 한국어 음성-텍스트(Inner Monologue) 정밀 정렬 엔진 및 외부 음성 모델 빌더 모듈."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from libs.codec import FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, resample_wav

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """단어 또는 음절 조각의 시작·종료 시각(초)을 표현하는 데이터클래스입니다."""

    text: str
    start_sec: float
    end_sec: float


AlignmentTimestampItem = Union[
    float,
    int,
    Tuple[str, float, float],
    Tuple[float, float],
    WordTimestamp,
    Dict[str, Any],
]
KoAlignFn = Callable[[str, torch.Tensor, Sequence[int]], Sequence[AlignmentTimestampItem]]


def _compute_subword_weights(
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
) -> List[float]:
    """각 서브워드 토큰의 공백 제거 글자 수에 비례하는 상대 발화 길이 가중치를 계산합니다."""
    if token_pieces is not None and len(token_pieces) == len(ko_token_ids):
        weights: List[float] = []
        for piece in token_pieces:
            cleaned = piece.replace("\u2581", "").strip()
            weights.append(float(max(1, len(cleaned))))
        return weights
    return [1.0] * len(ko_token_ids)


def align_tokens_by_acoustic_energy(
    ko_wav_24k: torch.Tensor,
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
    silence_db: float = -35.0,
    energy_gamma: float = 0.6,
) -> List[int]:
    """프레임별 RMS 에너지 포락선과 적응형 VAD를 이용해 실제 유성음 구간에 토큰 시작 프레임을 정렬합니다."""
    num_tokens = len(ko_token_ids)
    if num_tokens == 0:
        return []

    wav = ko_wav_24k.detach().cpu().float().reshape(-1)
    dur_frames = max(1, (wav.numel() + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
    if dur_frames == 1:
        return [0] * num_tokens

    padded_len = dur_frames * SAMPLES_PER_FRAME
    if wav.numel() < padded_len:
        wav_padded = torch.zeros(padded_len, dtype=torch.float32)
        wav_padded[: wav.numel()] = wav
    else:
        wav_padded = wav[:padded_len]

    frames = wav_padded.view(dur_frames, SAMPLES_PER_FRAME)
    rms = torch.sqrt(frames.pow(2).mean(dim=-1) + 1e-12)
    peak_rms = float(rms.max().item())

    if peak_rms <= 1e-6:
        return [(idx * dur_frames) // num_tokens for idx in range(num_tokens)]

    vad_thresh = peak_rms * (10.0 ** (silence_db / 20.0))
    voiced_indices = torch.nonzero(rms >= vad_thresh, as_tuple=False).reshape(-1)
    if voiced_indices.numel() > 0:
        onset_frame = int(voiced_indices[0].item())
        offset_frame = int(voiced_indices[-1].item())
    else:
        onset_frame = 0
        offset_frame = dur_frames - 1

    active_rms = rms[onset_frame : offset_frame + 1] / peak_rms
    activity = torch.where(
        active_rms >= (10.0 ** (silence_db / 20.0)),
        active_rms.clamp_min(1e-4).pow(energy_gamma),
        torch.full_like(active_rms, 0.015),
    )
    total_act = float(activity.sum().item())
    if total_act <= 1e-8:
        active_len = offset_frame - onset_frame + 1
        return [onset_frame + (idx * active_len) // num_tokens for idx in range(num_tokens)]

    cum_end = torch.cumsum(activity, dim=0) / total_act
    cum_start = torch.cat([torch.zeros(1, dtype=cum_end.dtype), cum_end[:-1]], dim=0)

    tok_weights = _compute_subword_weights(ko_token_ids, token_pieces)
    total_tok_w = sum(tok_weights)
    rel_frames: List[int] = []
    acc_w = 0.0
    for w in tok_weights:
        target_u = acc_w / max(1e-8, total_tok_w)
        cand = int(
            torch.searchsorted(
                cum_end, torch.tensor(target_u, dtype=cum_end.dtype), right=True
            ).item()
        )
        cand = max(0, min(cand, int(activity.numel()) - 1))
        if cand > 0 and float(cum_start[cand].item()) > target_u + 1e-6:
            cand -= 1
        rel_frames.append(onset_frame + cand)
        acc_w += w
    return rel_frames


def _parse_timestamp_items(
    raw_items: Sequence[AlignmentTimestampItem],
    max_sec: float,
) -> Tuple[Optional[List[float]], List[WordTimestamp]]:
    """외부 정렬기 반환값을 토큰별 시작 시각 리스트 또는 WordTimestamp 리스트로 정규화합니다."""
    if not raw_items:
        return None, []

    numeric_onsets: List[float] = []
    all_numeric = True
    for elem in raw_items:
        if isinstance(elem, (int, float)) and not isinstance(elem, bool):
            numeric_onsets.append(max(0.0, min(float(elem), max_sec)))
        else:
            all_numeric = False
            break
    if all_numeric:
        return numeric_onsets, []

    spans: List[WordTimestamp] = []
    for item in raw_items:
        w_text = ""
        s_sec: Optional[float] = None
        e_sec: Optional[float] = None
        if isinstance(item, WordTimestamp):
            w_text, s_sec, e_sec = item.text, item.start_sec, item.end_sec
        elif isinstance(item, dict):
            w_text = str(item.get("text") or item.get("word") or "")
            ts = item.get("timestamp") or item.get("timestamps")
            if isinstance(ts, (tuple, list)) and len(ts) >= 2:
                s_sec = float(ts[0]) if ts[0] is not None else None
                e_sec = float(ts[1]) if ts[1] is not None else None
            else:
                if item.get("start") is not None:
                    s_sec = float(item["start"])
                if item.get("end") is not None:
                    e_sec = float(item["end"])
        elif isinstance(item, (tuple, list)):
            if len(item) >= 3:
                w_text = str(item[0])
                s_sec = float(item[1]) if item[1] is not None else None
                e_sec = float(item[2]) if item[2] is not None else None
            elif len(item) == 2:
                s_sec = float(item[0]) if item[0] is not None else None
                e_sec = float(item[1]) if item[1] is not None else None

        if s_sec is None and e_sec is None:
            continue
        resolved_start = max(0.0, (float(e_sec) - 0.08) if s_sec is None else s_sec)
        resolved_end = (
            min(max_sec, resolved_start + 0.08)
            if (e_sec is None or e_sec < resolved_start)
            else e_sec
        )
        s_clamped = max(0.0, min(resolved_start, max_sec))
        e_clamped = max(s_clamped, min(resolved_end, max_sec))
        spans.append(WordTimestamp(text=w_text.strip(), start_sec=s_clamped, end_sec=e_clamped))

    return None, spans


def align_tokens_with_timestamps(
    raw_timestamps: Sequence[AlignmentTimestampItem],
    ko_wav_24k: torch.Tensor,
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
) -> List[int]:
    """단어·음절 타임스탬프 구간을 받아 각 한국어 서브워드 토큰의 0-기반 발화 시작 프레임을 계산합니다."""
    num_tokens = len(ko_token_ids)
    if num_tokens == 0:
        return []

    dur_samples = int(ko_wav_24k.numel())
    dur_frames = max(1, (dur_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
    max_sec = max(dur_samples / float(SAMPLE_RATE), dur_frames / FRAME_RATE)

    per_tok_onsets, word_spans = _parse_timestamp_items(raw_timestamps, max_sec=max_sec)
    if per_tok_onsets is not None and len(per_tok_onsets) == num_tokens:
        return [
            max(0, min(int(sec * FRAME_RATE), dur_frames - 1))
            for sec in per_tok_onsets
        ]

    if not word_spans:
        return align_tokens_by_acoustic_energy(
            ko_wav_24k, ko_token_ids, token_pieces=token_pieces
        )

    tok_weights = _compute_subword_weights(ko_token_ids, token_pieces)
    total_tok_w = max(1e-8, sum(tok_weights))

    span_char_weights = [
        float(max(1, len("".join(sp.text.split())))) for sp in word_spans
    ]
    total_span_w = max(1e-8, sum(span_char_weights))

    span_cum_end: List[float] = []
    acc_sw = 0.0
    for sw in span_char_weights:
        acc_sw += sw
        span_cum_end.append(acc_sw / total_span_w)

    rel_frames: List[int] = []
    acc_tw = 0.0
    span_idx = 0
    for tw in tok_weights:
        u = acc_tw / total_tok_w
        while span_idx + 1 < len(word_spans) and u >= span_cum_end[span_idx] - 1e-7:
            span_idx += 1
        prev_u = 0.0 if span_idx == 0 else span_cum_end[span_idx - 1]
        span_u_width = max(1e-6, span_cum_end[span_idx] - prev_u)
        local_ratio = max(0.0, min(1.0, (u - prev_u) / span_u_width))
        sp = word_spans[span_idx]
        onset_sec = sp.start_sec + local_ratio * max(0.0, sp.end_sec - sp.start_sec)
        f_idx = max(0, min(int(onset_sec * FRAME_RATE), dur_frames - 1))
        rel_frames.append(f_idx)
        acc_tw += tw

    return rel_frames


def compute_aligned_token_placement(
    ko_text: str,
    ko_wav_24k: torch.Tensor,
    ko_token_ids: Sequence[int],
    en_end_frame: int,
    prev_audio_end_frame: int,
    last_token_frame: int,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    token_pieces: Optional[Sequence[str]] = None,
    monologue_lead_frames: int = 1,
) -> Tuple[int, List[Tuple[int, int]], int]:
    """인과 제약과 선행 오프셋을 만족하도록 타깃 오디오 시작 샘플과 텍스트 토큰 프레임 위치를 결정합니다."""
    if align_mode not in ("auto", "energy", "whisper", "ctc", "timestamps", "uniform"):
        raise ValueError(
            f"지원하지 않는 align_mode입니다: {align_mode} "
            "(지원: 'auto', 'energy', 'whisper', 'ctc', 'timestamps', 'uniform')"
        )

    audio_floor = max(en_end_frame, prev_audio_end_frame)
    text_start_frame = max(audio_floor, last_token_frame + 1)
    dur_frames = max(
        1, (int(ko_wav_24k.numel()) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    )
    num_tokens = len(ko_token_ids)

    if num_tokens == 0:
        return audio_floor * SAMPLES_PER_FRAME, [], last_token_frame

    if align_mode == "uniform":
        lead_frames = max(0, num_tokens - dur_frames)
        audio_start_frame = text_start_frame + lead_frames
        token_positions: List[Tuple[int, int]] = []
        cur_last = last_token_frame
        for idx, tid in enumerate(ko_token_ids):
            if num_tokens <= dur_frames:
                ideal_pos = audio_start_frame + (idx * dur_frames) // max(1, num_tokens)
                f_pos = max(ideal_pos, cur_last + 1)
            else:
                f_pos = text_start_frame + idx
            token_positions.append((f_pos, int(tid)))
            cur_last = f_pos
        return audio_start_frame * SAMPLES_PER_FRAME, token_positions, cur_last

    use_ts = ko_align_fn is not None and align_mode in (
        "auto",
        "whisper",
        "ctc",
        "timestamps",
    )
    rel_acoustic: Optional[List[int]] = None
    if use_ts:
        assert ko_align_fn is not None
        try:
            raw_ts = ko_align_fn(ko_text, ko_wav_24k, ko_token_ids)
            rel_acoustic = align_tokens_with_timestamps(
                raw_ts, ko_wav_24k, ko_token_ids, token_pieces=token_pieces
            )
        except Exception as exc:
            logger.warning("외부 타임스탬프 정렬 실패로 에너지 정렬로 폴백합니다: %s", exc)
            rel_acoustic = None

    if rel_acoustic is None:
        rel_acoustic = align_tokens_by_acoustic_energy(
            ko_wav_24k, ko_token_ids, token_pieces=token_pieces
        )

    lead = max(0, int(monologue_lead_frames))
    rel_tok: List[int] = []
    for idx, r_f in enumerate(rel_acoustic):
        desired = int(r_f) - lead
        if idx == 0:
            rel_tok.append(desired)
        else:
            rel_tok.append(max(desired, rel_tok[-1] + 1))

    max_allowed_rel = max(dur_frames - 1, num_tokens - 1 - lead)
    overflow = rel_tok[-1] - max_allowed_rel
    if overflow > 0:
        rel_tok[-1] -= overflow
        for idx in range(num_tokens - 2, -1, -1):
            if rel_tok[idx] >= rel_tok[idx + 1]:
                rel_tok[idx] = rel_tok[idx + 1] - 1

    if num_tokens > dur_frames and rel_tok[0] > -(num_tokens - dur_frames):
        shift = rel_tok[0] + (num_tokens - dur_frames)
        rel_tok = [rt - shift for rt in rel_tok]

    audio_start_frame = max(audio_floor, text_start_frame - rel_tok[0])
    token_positions = []
    cur_last = last_token_frame
    for rt, tid in zip(rel_tok, ko_token_ids):
        f_pos = max(audio_start_frame + rt, cur_last + 1)
        token_positions.append((f_pos, int(tid)))
        cur_last = f_pos

    return audio_start_frame * SAMPLES_PER_FRAME, token_positions, cur_last


def build_hf_wavlm_sim_bin_fn(
    model_id: str,
    device: torch.device,
    num_speaker_sim_bins: int = 5,
) -> Callable[[torch.Tensor, torch.Tensor], int]:
    """HuggingFace WavLM X-Vector 모델로 화자 코사인 유사도를 이산 구간 인덱스로 양자화하는 함수를 생성합니다."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    feat_ext = AutoFeatureExtractor.from_pretrained(model_id)
    sv_model = AutoModelForAudioXVector.from_pretrained(model_id).to(device).eval()
    target_sr = int(getattr(feat_ext, "sampling_rate", 16000))

    def _embed(wav_24k: torch.Tensor) -> torch.Tensor:
        """24kHz 파형에서 정규화된 화자 임베딩 벡터를 추출합니다."""
        w = wav_24k.detach().cpu().float().reshape(-1)
        if w.numel() < SAMPLE_RATE // 4:
            w = F.pad(w, (0, SAMPLE_RATE // 4 - w.numel()))
        w_16k = resample_wav(w, SAMPLE_RATE, target_sr)
        inputs = feat_ext(
            w_16k.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = sv_model(**inputs).embeddings[0]
        return F.normalize(emb.float(), dim=-1)

    def _score_bin(prompt_24k: torch.Tensor, gen_24k: torch.Tensor) -> int:
        """원문 화자 프롬프트와 타깃 발화 간의 코사인 유사도를 구간 번호로 변환합니다."""
        e1 = _embed(prompt_24k)
        e2 = _embed(gen_24k)
        cos_sim = float(torch.sum(e1 * e2).item())
        norm_score = (cos_sim - 0.50) / 0.40
        bin_idx = int(norm_score * num_speaker_sim_bins)
        return max(0, min(num_speaker_sim_bins - 1, bin_idx))

    return _score_bin


def build_hf_whisper_align_fn(
    model_id: str,
    device: torch.device,
) -> KoAlignFn:
    """HuggingFace Whisper 단어 타임스탬프 파이프라인 기반 한국어 정렬 함수를 생성합니다."""
    from transformers import pipeline

    pipe_device = 0 if device.type == "cuda" else -1
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=pipe_device,
    )

    def _align_fn(
        ko_text: str,
        ko_wav_24k: torch.Tensor,
        ko_token_ids: Sequence[int],
    ) -> Sequence[AlignmentTimestampItem]:
        """오디오에서 단어 단위 시작·종료 시각 리스트를 추출합니다."""
        del ko_text, ko_token_ids
        w_16k = resample_wav(
            ko_wav_24k.detach().cpu().float().reshape(-1), SAMPLE_RATE, 16000
        )
        result = asr_pipe(
            w_16k.numpy(),
            return_timestamps="word",
            generate_kwargs={"language": "korean", "task": "transcribe"},
        )
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        out: List[WordTimestamp] = []
        max_sec = float(ko_wav_24k.numel()) / float(SAMPLE_RATE)
        for ch in chunks:
            ts = ch.get("timestamp")
            txt = str(ch.get("text") or "").strip()
            if not isinstance(ts, (tuple, list)) or len(ts) < 2:
                continue
            s_sec = float(ts[0]) if ts[0] is not None else 0.0
            e_sec = float(ts[1]) if ts[1] is not None else max_sec
            out.append(WordTimestamp(text=txt, start_sec=s_sec, end_sec=max(s_sec, e_sec)))
        return out

    return _align_fn


def _viterbi_ctc_forced_align(
    log_probs: torch.Tensor,
    target_ids: List[int],
    blank_id: int,
) -> List[Tuple[int, int]]:
    """벡터화된 PyTorch Viterbi 동적 계획법으로 CTC 로그 확률 행렬에 정답 토큰 시퀀스를 강제 정렬합니다."""
    num_frames = int(log_probs.shape[0])
    num_targets = len(target_ids)
    if num_targets == 0 or num_frames == 0:
        return []

    ext_tokens: List[int] = [blank_id]
    for tid in target_ids:
        ext_tokens.extend([tid, blank_id])
    num_states = len(ext_tokens)

    neg_inf = -1e30
    lp_cpu = log_probs.detach().cpu().float()
    ext_idx = torch.tensor(ext_tokens, dtype=torch.long)
    emit_scores = lp_cpu[:, ext_idx]

    dp = torch.full((num_frames, num_states), neg_inf, dtype=torch.float32)
    backptr = torch.zeros((num_frames, num_states), dtype=torch.long)

    dp[0, 0] = emit_scores[0, 0]
    if num_states > 1:
        dp[0, 1] = emit_scores[0, 1]

    can_skip2 = torch.zeros(num_states, dtype=torch.bool)
    if num_states > 2:
        can_skip2[2:] = (ext_idx[2:] != blank_id) & (ext_idx[2:] != ext_idx[:-2])

    s_arange = torch.arange(num_states, dtype=torch.long)
    neg_inf_1 = torch.full((1,), neg_inf, dtype=torch.float32)
    neg_inf_2 = torch.full((min(2, num_states),), neg_inf, dtype=torch.float32)
    neg_inf_s = torch.full((num_states,), neg_inf, dtype=torch.float32)

    for t in range(1, num_frames):
        prev = dp[t - 1]
        c0 = prev
        c1 = torch.cat([neg_inf_1, prev[:-1]], dim=0) if num_states > 1 else neg_inf_s
        if num_states > 2:
            c2_raw = torch.cat([neg_inf_2, prev[:-2]], dim=0)
            c2 = torch.where(can_skip2, c2_raw, neg_inf_s)
        else:
            c2 = neg_inf_s
        cands = torch.stack([c0, c1, c2], dim=0)
        best_vals, best_offsets = cands.max(dim=0)
        dp[t] = best_vals + emit_scores[t]
        backptr[t] = s_arange - best_offsets

    last_s = num_states - 1
    if num_states >= 2 and float(dp[-1, num_states - 2].item()) > float(
        dp[-1, num_states - 1].item()
    ):
        last_s = num_states - 2

    backptr_list = backptr.tolist()
    state_path = [0] * num_frames
    cur_s = last_s
    for t in range(num_frames - 1, -1, -1):
        state_path[t] = cur_s
        cur_s = backptr_list[t][cur_s]

    spans: List[Tuple[int, int]] = []
    for tok_i in range(num_targets):
        target_state = 2 * tok_i + 1
        frames_for_tok = [t for t, s in enumerate(state_path) if s == target_state]
        if frames_for_tok:
            spans.append((frames_for_tok[0], frames_for_tok[-1] + 1))
        else:
            approx = (tok_i * num_frames) // max(1, num_targets)
            spans.append((approx, min(num_frames, approx + 1)))
    return spans


def build_hf_ctc_forced_align_fn(
    model_id: str,
    device: torch.device,
) -> KoAlignFn:
    """한국어 Wav2Vec2/MMS CTC 모델과 Viterbi 강제 정렬을 사용하여 음절별 시각을 추출하는 함수를 생성합니다."""
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    ctc_model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    vocab_values = set(processor.tokenizer.get_vocab().values())
    blank_id = int(getattr(ctc_model.config, "pad_token_id", 0))

    def _align_fn(
        ko_text: str,
        ko_wav_24k: torch.Tensor,
        ko_token_ids: Sequence[int],
    ) -> Sequence[AlignmentTimestampItem]:
        """정답 한국어 문자열을 오디오 프레임에 강제 정렬하여 음절 타임스탬프를 반환합니다."""
        del ko_token_ids
        w_16k = resample_wav(
            ko_wav_24k.detach().cpu().float().reshape(-1), SAMPLE_RATE, target_sr
        )
        if w_16k.numel() < target_sr // 10:
            return []
        inputs = processor(w_16k.numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = ctc_model(input_values).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)

        total_dur_sec = float(ko_wav_24k.numel()) / float(SAMPLE_RATE)
        num_ctc_frames = max(1, int(log_probs.shape[0]))
        sec_per_frame = total_dur_sec / float(num_ctc_frames)

        encoded_chars: List[int] = []
        valid_chars: List[str] = []
        for ch in ko_text:
            if ch.isspace():
                continue
            tok_ids = processor.tokenizer.encode(ch, add_special_tokens=False)
            tok_ids = [tid for tid in tok_ids if tid != blank_id and tid in vocab_values]
            if tok_ids:
                encoded_chars.extend(tok_ids)
                valid_chars.extend([ch] * len(tok_ids))

        if not encoded_chars or num_ctc_frames < len(encoded_chars):
            return []

        spans = _viterbi_ctc_forced_align(log_probs, encoded_chars, blank_id=blank_id)
        return [
            WordTimestamp(
                text=ch,
                start_sec=s_f * sec_per_frame,
                end_sec=max(s_f * sec_per_frame, e_f * sec_per_frame),
            )
            for ch, (s_f, e_f) in zip(valid_chars, spans)
        ]

    return _align_fn
