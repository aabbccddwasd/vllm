# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the padding a grouped MoE layout pays for.

A grouped MoE layout pads every expert slot to the alignment, so the GEMM sees
``M_sum`` rows where only ``M * topk`` are real work. Both report shapes read
that back from the live batch rather than inferring it from an ncu profile. They
are no-ops unless ``VLLM_MOE_PAD_LOG=1``, and they read device memory, so they
skip (and report) CUDA-graph capture, where a synchronization is not allowed.

Usage::

    note_moe_padding(M=M, num_topk=topk, ..., expert_tokens_meta=meta)
    note_moe_layout(M=M, num_topk=topk, ..., expert_start_loc=loc)
"""

from __future__ import annotations

import torch

from vllm.profiler.instrumentation import enabled, note_once


def note_moe_padding(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment_in: int,
    alignment_used: int,
    M_sum: int,
    expert_tokens_meta,
) -> None:
    """Report the (M, alignment, M_sum, active experts) of one grouped GEMM."""
    if not enabled("moe_pad"):
        return
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    useful = M * num_topk
    padding = "n/a (no device read during capture)"
    if not capturing and expert_tokens_meta is not None:
        expert_num_tokens = getattr(expert_tokens_meta, "expert_num_tokens", None)
        if expert_num_tokens is not None:
            try:
                active = int((expert_num_tokens > 0).sum().item())
                rows = (
                    (expert_num_tokens.to(torch.int64) + alignment_used - 1)
                    // alignment_used
                    * alignment_used
                )
                exact_m_sum = int(rows.sum().item())
                padding = (
                    f"active_experts={active} exact_M_sum={exact_m_sum} "
                    f"useful={useful} upper_bound={M_sum} "
                    f"padding={100 * (1 - useful / max(exact_m_sum, 1)):.1f}%"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics are never fatal
                padding = f"device read failed: {type(exc).__name__}"
    note_once(
        "moe_pad",
        "padding",
        M=M,
        topk=num_topk,
        experts=local_num_experts,
        alignment=f"{alignment_in}->{alignment_used}",
        M_sum=M_sum,
        useful=useful,
        capturing=capturing,
        detail=padding,
    )


def note_moe_layout(
    M: int,
    num_topk: int,
    local_num_experts: int,
    align_used: int,
    M_sum: int,
    expert_num_tokens,
    expert_start_loc,
) -> None:
    """Report the slots each expert occupies and the padding they carry.

    ``expert_start_loc`` differences are the rows an expert actually occupies
    (its aligned slot), ``expert_num_tokens`` the real token count; the gap is
    the padding. Skipped during capture, which allows no device read.
    """
    if not enabled("moe_pad"):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    try:
        num_tokens = expert_num_tokens.to(torch.int64).cpu()
        start_loc = expert_start_loc.to(torch.int64).cpu()
        # ``expert_start_loc`` holds one entry per expert, so its differences
        # only cover the experts that have a successor.
        slots = (start_loc[1:] - start_loc[:-1])[num_tokens[:-1] > 0]
        active = int((num_tokens > 0).sum())
        useful = int(num_tokens.sum())
        used_rows = int(slots.sum())
        note_once(
            "moe_pad",
            "layout",
            M=M,
            topk=num_topk,
            experts=local_num_experts,
            align_used=align_used,
            M_sum=M_sum,
            active_experts=active,
            useful_rows=useful,
            used_rows=used_rows,
            slot_min=int(slots.min()) if slots.numel() else 0,
            slot_max=int(slots.max()) if slots.numel() else 0,
            padding=f"{100 * (1 - useful / max(used_rows, 1)):.1f}%",
            rows_per_M_sum=f"{used_rows / max(M_sum, 1):.3f}",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics are never fatal
        note_once("moe_pad", "layout_read_failed", error=type(exc).__name__)
