import math
import torch
import torch.nn as nn
from EDarch_from_scratch import EDarchModel, LowRankScaledEmbedding


def adapt_EDarch_for_korean_vocab(
    model: EDarchModel,
    new_ko_vocab_size: int,
    preserve_special_tokens: bool = True,
) -> EDarchModel:
    cfg = model.cfg
    min_vocab_size = max(cfg.text_eos_id, cfg.text_pad_id) + 1
    if new_ko_vocab_size < min_vocab_size:
        raise ValueError(
            f"new_ko_vocab_size({new_ko_vocab_size})는 특수 토큰 ID(EOS={cfg.text_eos_id}, "
            f"PAD={cfg.text_pad_id})를 포함할 수 있도록 최소 {min_vocab_size} 이상이어야 합니다."
        )

    old_bos_id = cfg.text_bos_id
    old_text_emb = model.text_emb
    old_text_linear = model.text_linear
    old_dep_text_emb = model.depformer_text_emb

    cfg.text_card = new_ko_vocab_size
    new_bos_id = cfg.text_bos_id

    device = old_text_emb.emb.weight.device
    dtype = old_text_emb.emb.weight.dtype

    new_text_emb = LowRankScaledEmbedding(
        new_ko_vocab_size + 1, cfg.dim, zero_idx=cfg.zero_token_id
    ).to(device=device, dtype=dtype)

    new_text_linear = nn.Linear(cfg.dim, new_ko_vocab_size, bias=False).to(
        device=device, dtype=dtype
    )

    new_dep_text_emb = LowRankScaledEmbedding(
        new_ko_vocab_size + 1,
        cfg.depformer_dim,
        zero_idx=cfg.zero_token_id,
        low_rank=cfg.depformer_low_rank_emb,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        for new_w, old_w in (
            (new_text_emb.emb.weight, old_text_emb.emb.weight),
            (new_text_linear.weight, old_text_linear.weight),
            (new_dep_text_emb.emb.weight, old_dep_text_emb.emb.weight),
        ):
            w_f = old_w.float()
            mean = float(w_f.mean().item())
            raw_std = (
                float(w_f.std(correction=0).item())
                if w_f.numel() <= 1
                else float(w_f.std().item())
            )
            std = 1e-4 if (math.isnan(raw_std) or raw_std < 1e-4) else raw_std
            nn.init.normal_(new_w, mean=mean, std=std)

    if preserve_special_tokens:
        with torch.no_grad():
            for tok_id in (cfg.text_eos_id, cfg.text_pad_id):
                if tok_id < old_text_emb.emb.num_embeddings:
                    new_text_emb.emb.weight[tok_id].copy_(old_text_emb.emb.weight[tok_id])
                if tok_id < old_dep_text_emb.emb.num_embeddings:
                    new_dep_text_emb.emb.weight[tok_id].copy_(old_dep_text_emb.emb.weight[tok_id])
                if tok_id < old_text_linear.out_features:
                    new_text_linear.weight[tok_id].copy_(old_text_linear.weight[tok_id])

            if old_bos_id < old_text_emb.emb.num_embeddings:
                new_text_emb.emb.weight[new_bos_id].copy_(old_text_emb.emb.weight[old_bos_id])
            if old_bos_id < old_dep_text_emb.emb.num_embeddings:
                new_dep_text_emb.emb.weight[new_bos_id].copy_(
                    old_dep_text_emb.emb.weight[old_bos_id]
                )

    if (
        old_dep_text_emb.low_rank_proj is not None
        and new_dep_text_emb.low_rank_proj is not None
    ):
        with torch.no_grad():
            new_dep_text_emb.low_rank_proj.weight.copy_(
                old_dep_text_emb.low_rank_proj.weight
            )

    model.text_emb = new_text_emb
    model.text_linear = new_text_linear
    model.depformer_text_emb = new_dep_text_emb

    return model
