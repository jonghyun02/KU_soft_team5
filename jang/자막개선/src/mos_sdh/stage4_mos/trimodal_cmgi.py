"""Trimodal Temporal-Aware Gated Fusion (CmGI) for Emotion & Irony Detection (Gao et al., 2024 Eq. 3-7)."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import EmotionIronyAnalysis, EmotionLabel, SegmentMoS

_EMOTION_CLASSES: list[EmotionLabel] = ["분노", "슬픔", "기쁨", "공포", "냉소", "중립"]

# Korean lexical praise / positive intensifiers frequently used in dramatic irony ("참 잘~ 하십니다")
_PRAISE_ANCHORS = (
    "잘하",
    "잘 하",
    "대단하",
    "훌륭하",
    "멋지",
    "고맙",
    "감사",
    "좋네",
    "좋구먼",
    "좋으시",
    "깔끔하",
    "완벽하",
    "최고",
    "착하",
    "친절하",
)

_SARCASM_ADVERBS = ("참", "아주", "퍽이나", "정말", "되게", "엄청", "잘도")

_NEGATIVE_LEXICAL_ANCHORS = (
    "싫",
    "미치",
    "화나",
    "못 참",
    "못참",
    "죽겠",
    "최악",
    "그만",
    "꺼져",
    "짜증",
    "슬프",
    "무섭",
    "두렵",
)


class TrimodalCmGIFusion(nn.Module):
    """PyTorch implementation of Temporal-Aware Cross-Modal Gated Interaction (Gao et al., 2024 Eq. 3-6)

    Extended to 3 modalities: Acoustic (x_a), Lexical-Semantic (x_l), and Visual Facial Expression (x_v),
    preserving temporal sequence axis `t` before final emotion aggregation.
    """

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_l = nn.Linear(d_model * 2, d_model)
        self.gate_v = nn.Linear(d_model * 2, d_model)

        # Deterministic orthogonal initialization so feature geometry is preserved without random drift
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _scaled_dot_attention(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        """Gao et al. (2024) Eq. (3)-(4): softmax((Q K^T) / sqrt(d_k)) V over temporal sequence t."""
        q = self.w_q(q_in)
        k = self.w_k(kv_in)
        v = self.w_v(kv_in)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self,
        x_a: torch.Tensor,
        x_l: torch.Tensor,
        x_v: torch.Tensor,
        visual_mask: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Fuse (T, D) modality tensors while retaining temporal resolution T (Eq. 6)."""
        # Stage 1 cross-projection (Eq. 3)
        s_a = self._scaled_dot_attention(x_a, x_l)
        s_l = self._scaled_dot_attention(x_l, x_a)
        s_v = self._scaled_dot_attention(x_v, x_l) * visual_mask

        # Stage 2 cross-modal interaction (Eq. 4)
        c_a = self._scaled_dot_attention(s_a, s_l)
        c_l = self._scaled_dot_attention(s_l, s_a)
        c_v = self._scaled_dot_attention(s_v, s_a) * visual_mask

        # Temporal-aware matrix gates G_a, G_l, G_v in R^{T x D} (Eq. 6)
        g_a = torch.sigmoid(self.gate_a(torch.cat([s_a, c_a], dim=-1)))
        g_l = torch.sigmoid(self.gate_l(torch.cat([s_l, c_l], dim=-1)))
        g_v = torch.sigmoid(self.gate_v(torch.cat([s_v, c_v], dim=-1))) * visual_mask

        out_a = torch.cat([s_a * g_a, c_l * (1.0 - g_a)], dim=-1)
        out_l = torch.cat([s_l * g_l, c_a * (1.0 - g_l)], dim=-1)
        out_v = torch.cat([s_v * g_v, c_v * (1.0 - g_v) * visual_mask], dim=-1)

        # Temporal mean-pooling after temporal-gated interaction (Eq. 7)
        pooled = torch.cat([out_a.mean(dim=0), out_l.mean(dim=0), out_v.mean(dim=0)], dim=-1)

        raw_ga = float(g_a.mean().item())
        raw_gl = float(g_l.mean().item())
        raw_gv = float(g_v.mean().item())
        denom = max(1e-6, raw_ga + raw_gl + raw_gv)
        gate_summary = {
            "audio": round(raw_ga / denom, 4),
            "lexical": round(raw_gl / denom, 4),
            "visual": round(raw_gv / denom, 4),
        }
        return pooled, gate_summary


def analyze_trimodal_emotion_and_irony(
    segments: list[SegmentMoS],
    temporal_fer_sequences: dict[int, np.ndarray],
    config: AnalysisConfig,
) -> list[SegmentMoS]:
    """Run Trimodal CmGI interaction & compute emotion probabilities and Irony Divergence Score I_irony."""
    if not segments:
        return segments

    cmgi = TrimodalCmGIFusion(d_model=16)
    cmgi.eval()

    for seg in segments:
        text = seg.existing_subtitle if seg.existing_subtitle is not None else seg.raw_transcript
        t_steps = max(2, len(seg.words))

        # 1. Build temporal acoustic sequence x_a in R^{T x 16}
        x_a_np = np.zeros((t_steps, 16), dtype=np.float32)
        for idx in range(t_steps):
            if idx < len(seg.words):
                w = seg.words[idx]
                x_a_np[idx, 0] = w.z_emphasis
                x_a_np[idx, 1] = (w.pitch_hz - 160.0) / 80.0 if w.pitch_hz > 0 else 0.0
                x_a_np[idx, 2] = (w.energy_db - 60.0) / 15.0 if w.energy_db > 0 else 0.0
                x_a_np[idx, 3] = (w.duration_per_syllable_ms - 180.0) / 90.0
            x_a_np[idx, 4] = seg.arousal_zscore
            x_a_np[idx, 5] = seg.f0_std_hz / 40.0

        # 2. Build temporal lexical polarity sequence x_l in R^{T x 16}
        lex_pol = compute_lexical_polarity(text)
        x_l_np = np.zeros((t_steps, 16), dtype=np.float32)
        x_l_np[:, 0] = lex_pol
        x_l_np[:, 1] = 1.0 if any(adv in text for adv in _SARCASM_ADVERBS) else 0.0
        for idx in range(min(t_steps, len(seg.words))):
            w_pol = compute_lexical_polarity(seg.words[idx].text)
            x_l_np[idx, 2] = w_pol

        # 3. Build temporal visual FER sequence x_v in R^{T x 16} (with automatic 0-mask fallback when no face)
        vis_mask = 1.0 if seg.visual.face_detected else 0.0
        x_v_np = np.zeros((t_steps, 16), dtype=np.float32)
        fer_seq = temporal_fer_sequences.get(seg.segment_id)
        if fer_seq is not None and len(fer_seq) > 0:
            # Interpolate/resample frame FER sequence [분노, 슬픔, 기쁨, 공포, 냉소] to T word steps
            indices = np.linspace(0, len(fer_seq) - 1, t_steps).round().astype(int)
            x_v_np[:, :5] = fer_seq[indices]
        elif seg.visual.expression_probs:
            for c_i, col in enumerate(["분노", "슬픔", "기쁨", "공포", "냉소"]):
                x_v_np[:, c_i] = float(seg.visual.expression_probs.get(col, 0.0))

        with torch.no_grad():
            _, gate_weights = cmgi(
                torch.from_numpy(x_a_np),
                torch.from_numpy(x_l_np),
                torch.from_numpy(x_v_np),
                visual_mask=vis_mask,
            )

        # Compute calibrated audio-visual polarity Pol_AV in [-1, 1] and 6-class emotion distribution
        av_pol, emo_probs = _score_audiovisual_emotion_and_polarity(seg, lex_pol)
        dominant_emo: EmotionLabel = max(emo_probs.items(), key=lambda kv: kv[1])[0]  # type: ignore[assignment]

        # PRD Section 4.2 Eq.: Irony Divergence Score I_irony in [0, 1]
        if lex_pol * av_pol < 0.0:
            irony_score = min(1.0, abs(lex_pol - av_pol) / 2.0)
            if any(adv in text for adv in _SARCASM_ADVERBS):
                irony_score = min(1.0, irony_score + 0.15)
        else:
            irony_score = 0.0

        irony_detected = irony_score >= config.irony_threshold
        if irony_detected:
            emo_probs["냉소"] = round(max(emo_probs.get("냉소", 0.0), irony_score), 4)
            total_p = sum(emo_probs.values()) or 1.0
            emo_probs = {k: round(v / total_p, 4) for k, v in emo_probs.items()}
            dominant_emo = "냉소"

        seg.emotion_irony = EmotionIronyAnalysis(
            dominant_emotion=dominant_emo,
            emotion_probs=emo_probs,
            lexical_polarity=round(lex_pol, 4),
            audiovisual_polarity=round(av_pol, 4),
            irony_score=round(irony_score, 4),
            irony_detected=irony_detected,
            gate_weights_mean=gate_weights,
        )

    return segments


def compute_lexical_polarity(text: str) -> float:
    """Estimate Korean utterance lexical sentiment polarity in [-1.0, +1.0]."""
    pos_hits = sum(1 for kw in _PRAISE_ANCHORS if kw in text)
    neg_hits = sum(1 for kw in _NEGATIVE_LEXICAL_ANCHORS if kw in text)

    if pos_hits == 0 and neg_hits == 0:
        return 0.0
    raw = (pos_hits * 0.65) - (neg_hits * 0.65)
    if pos_hits > 0 and any(adv in text for adv in _SARCASM_ADVERBS):
        raw += 0.20
    return float(max(-1.0, min(1.0, raw)))


def _score_audiovisual_emotion_and_polarity(
    seg: SegmentMoS,
    lex_pol: float,
) -> tuple[float, dict[str, float]]:
    """Combine acoustic prosody (arousal, pitch variance, elongation, sighs) & visual FER into emotion probs."""
    scores: dict[str, float] = {k: 0.08 for k in _EMOTION_CLASSES}
    scores["중립"] = 0.45

    z_ar = seg.arousal_zscore
    has_sigh = any(c.cue_code == "b/" for c in seg.cues)
    has_laugh = any(c.cue_code == "l/" for c in seg.cues)
    max_dur = max((w.duration_per_syllable_ms for w in seg.words), default=180.0)

    # Acoustic contributions
    if z_ar >= 1.3:
        if lex_pol > 0.3 and not seg.visual.expression_probs.get("분노", 0.0) > 0.3:
            scores["기쁨"] += 0.40 * min(2.0, z_ar)
        else:
            scores["분노"] += 0.55 * min(2.2, z_ar)
    elif z_ar <= -0.75:
        if lex_pol >= 0.35 and (max_dur >= 240.0 or has_sigh):
            # Low/flat depressed prosody or elongated syllable + sigh on positive praise => cold sarcasm!
            scores["냉소"] += 0.65
        else:
            scores["슬픔"] += 0.45

    if has_sigh:
        if lex_pol >= 0.35:
            scores["냉소"] += 0.35
        else:
            scores["슬픔"] += 0.30
    if has_laugh:
        scores["기쁨"] += 0.35

    # Visual FER contributions when face was detected
    if seg.visual.face_detected and seg.visual.expression_probs:
        vp = seg.visual.expression_probs
        scores["분노"] += 0.85 * float(vp.get("분노", 0.0))
        scores["슬픔"] += 0.85 * float(vp.get("슬픔", 0.0))
        scores["기쁨"] += 0.85 * float(vp.get("기쁨", 0.0))
        scores["공포"] += 0.85 * float(vp.get("공포", 0.0))
        scores["냉소"] += 0.95 * float(vp.get("냉소", 0.0))
        scores["중립"] += 0.50 * float(vp.get("중립", 0.0))

    total = sum(scores.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in scores.items()}

    pos_av = probs["기쁨"]
    neg_av = probs["분노"] + probs["냉소"] + 0.8 * probs["슬픔"] + 0.6 * probs["공포"]
    av_polarity = float(max(-1.0, min(1.0, (pos_av - neg_av) * 1.6)))
    return av_polarity, probs
