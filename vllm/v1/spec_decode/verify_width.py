# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick how many draft tokens to verify from a cost model.

Speculative decoding trades rows for tokens: verifying one more draft position
adds one row to the verify batch, which costs ``c_row`` milliseconds and yields
``S_j`` accepted tokens in expectation, where ``S_j`` is the probability that the
first ``j`` drafts are all accepted. Which width is best therefore depends on how
much a token is worth at the current batch size, and that is set by the batch
itself:

    cost(widths) = t0 + c_row * (other_rows + sum_i (1 + widths_i))
                   + c_req * num_reqs                                 [ms]
    expected(widths) = sum_i (1 + sum_{j <= widths_i} S_j^{(i)})      [tokens]
    Theta = expected / cost                                           [tokens/ms]

A position is worth verifying iff its marginal tokens-per-ms beats the batch's
own average, i.e. iff ``S_j > c_row * Theta``. Both sides depend on the widths,
so the optimum is the fixed point of

    widths <- #{j : S_j > c_row * expected(widths) / cost(widths)}

which a few iterations reach from the full-width start. Solving it rather than
hard-coding a width is what makes the rule follow the load: with a measured
``t0 = 9.39`` ms and ``c_row = 0.592`` ms the optimum is a wide verify at one
request and falls to one or two positions by a dozen requests, because at a large
batch the step is already throughput-saturated and a marginal row buys less than
it costs.

The solve runs entirely on the device with a fixed iteration count and no
host synchronization, so it can sit in the async scheduling path without stalling
it; the caller takes a single device-to-host copy of the resulting widths.

``c_row`` is the marginal cost of verifying one more position, i.e. the per-token
verify cost. It is a property of the load, not of the hardware — 0.59 ms on a
single request, down to 0.11 ms on a saturated batch where the weights and expert
traffic are already paid for — so :class:`RowCostModel` fits it online from the
engine's own step times. The deployment supplies one number, that cost; the
threshold, the width and the rest of the batch's cost model are all derived.
"""

from __future__ import annotations

import torch

__all__ = [
    "solve_verify_widths",
    "expected_tokens",
    "step_cost_ms",
    "RowCostModel",
]

# Runs to convergence in practice; a fixed count keeps the solve sync-free.
DEFAULT_ITERS = 3

# How the engine measures its own cost curve. These are implementation constants,
# not decision knobs: the only thing a deployment specifies is the per-token
# verify cost (``VLLM_DSPARK_VERIFY_COST_MS``), which anchors the fit; the row
# price, the intercept and the ragged penalty are all measured from the steps the
# engine is already running, and the width follows from them.
COST_WINDOW_STEPS = 8
COST_INTERVAL_STEPS = 128
COST_FORGETTING = 0.98
COST_MIN_SAMPLES = 8
# 行斜率要在多大的行区间上才可信。单流（rows 1–6）区间太窄：步时噪声主导，
# 实测拟合把 c_row 抬到 2.3–3.2 ms/行（真实量级远小），于是规则以为"少验一行省很多"
# 而砍位。区间不够宽时不上拟合值，退回"实测截距 + 指定的每 token 验证开销"。
COST_MIN_ROW_SPREAD = 4


class RowCostModel:
    """Online estimate of step cost from the engine's own step times.

    The fit is a four-term linear model in milliseconds::

        t_step ~= t0 + c_row * rows + c_req * num_reqs + c_ragged * ragged

    The decision needs the marginal cost of one more verify row *at the current
    operating point*, and that is not a property of the hardware alone. Measured
    on a single request a row costs 0.59 ms and deepening the verify is cheap; on
    a saturated batch the weights and expert traffic are shared out and the next
    row is nearly free. On this machine the curve is not even linear: 0.43 ms per
    row around eight rows against 0.18 ms around twenty. A constant cannot
    express any of that, so the engine fits its own curve from the steps it is
    already running.

    The extra regressors exist to keep the row price honest (see the measured
    evidence in ``WIDTH-COST-MODEL.md``):

    * ``num_reqs`` — rows and requests move together whenever every request keeps
      the same width, and a two-parameter fit would fold the per-request costs
      (KV reads, sampling, the draft graph's own rows) into ``c_row``.
    * ``ragged`` — a batch whose requests verify different widths runs mixed
      query lengths, which costs extra per step. Left out, that penalty would be
      blamed on the rows and would make shallow splits look cheaper than they
      are.

    The fit is recursive least squares with exponential forgetting: one
    host-side multiply-add per sample and no device synchronization of its own.

    Args:
        t0: Prior/intercept in ms, also the fallback while unidentified.
        c_row: Prior per-row cost in ms.
        forgetting: Forgetting factor in (0, 1]; lower tracks load changes faster.
        min_samples: Samples required before the fit is trusted.
        min_row_spread: Row-count spread required before the fit is trusted.
            Without spread the slope is not identifiable, only the intercept is,
            and a fit that never saw two different row counts would report the
            prior back with false confidence.
        c_row_bounds: Clamp on the fitted slope; a non-positive or absurd slope
            is treated as not-yet-identified.

    """

    def __init__(
        self,
        t0: float,
        c_row: float,
        forgetting: float = 0.98,
        min_samples: int = 8,
        min_row_spread: int = 4,
        c_row_bounds: tuple[float, float] = (1e-4, 10.0),
    ) -> None:
        # [t0, c_row, c_req, c_ragged]
        self._w = [float(t0), float(c_row), 0.0, 0.0]
        # Prior covariance. The slope is far less certain than the intercept,
        # because a single batch size carries almost no information about it.
        self._p = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
        self._forgetting = forgetting
        self._min_samples = min_samples
        self._min_row_spread = min_row_spread
        self._bounds = c_row_bounds
        self._n = 0
        self._rows_min: int | None = None
        self._rows_max: int | None = None
        self._reqs_min: int | None = None
        self._reqs_max: int | None = None
        self._ragged_samples = 0

    @property
    def num_samples(self) -> int:
        return self._n

    @property
    def row_spread(self) -> int:
        if self._rows_min is None or self._rows_max is None:
            return 0
        return self._rows_max - self._rows_min

    def observe(
        self, rows: int, num_reqs: int, step_ms: float, ragged: bool = False
    ) -> None:
        """Fold one measured step into the fit."""
        if rows <= 0 or num_reqs <= 0 or not (step_ms > 0.0) or step_ms != step_ms:
            return
        self._n += 1
        self._ragged_samples += int(ragged)
        self._rows_min = rows if self._rows_min is None else min(self._rows_min, rows)
        self._rows_max = rows if self._rows_max is None else max(self._rows_max, rows)
        self._reqs_min = (
            num_reqs if self._reqs_min is None else min(self._reqs_min, num_reqs)
        )
        self._reqs_max = (
            num_reqs if self._reqs_max is None else max(self._reqs_max, num_reqs)
        )

        x = (1.0, float(rows), float(num_reqs), 1.0 if ragged else 0.0)
        dim = len(x)
        px = [sum(self._p[i][j] * x[j] for j in range(dim)) for i in range(dim)]
        denom = self._forgetting + sum(x[i] * px[i] for i in range(dim))
        gain = [px[i] / denom for i in range(dim)]
        error = step_ms - sum(x[i] * self._w[i] for i in range(dim))
        for i in range(dim):
            self._w[i] += gain[i] * error
        # P <- (P - k x^T P) / lambda.
        self._p = [
            [
                (
                    self._p[i][j]
                    - sum(gain[i] * x[m] * self._p[m][j] for m in range(dim))
                )
                / self._forgetting
                for j in range(dim)
            ]
            for i in range(dim)
        ]

    def intercept(self) -> tuple[float, float] | None:
        """``(t0, c_req)`` once there are enough samples, before the slope is known.

        The batch's fixed cost is measurable from steps of any one size, while the
        row price needs two sizes to be identified. Until then the deployment's
        specified per-token cost is the honest row price, and the decision can
        still run: only the slope is missing, not the model.
        """
        if self._n < self._min_samples:
            return None
        return max(0.0, self._w[0]), self._w[2]

    def estimate(self) -> tuple[float, float, float, float] | None:
        """The fitted ``(t0, c_row, c_req, c_ragged)``, or None if unidentified."""
        if self._n < self._min_samples or self.row_spread < self._min_row_spread:
            return None
        c_row = self._w[1]
        if not (self._bounds[0] <= c_row <= self._bounds[1]):
            return None
        return max(0.0, self._w[0]), c_row, self._w[2], self._w[3]

    def state(self) -> dict[str, float]:
        """Fit state for logging."""
        return {
            "n": float(self._n),
            "ragged": float(self._ragged_samples),
            "row_min": float(self._rows_min if self._rows_min is not None else 0),
            "row_max": float(self._rows_max if self._rows_max is not None else 0),
            "req_min": float(self._reqs_min if self._reqs_min is not None else 0),
            "req_max": float(self._reqs_max if self._reqs_max is not None else 0),
            "t0": self._w[0],
            "c_row": self._w[1],
            "c_req": self._w[2],
            "c_ragged": self._w[3],
        }


def step_cost_ms(rows: float, t0: float, c_row: float) -> float:
    """Wall time of one step with ``rows`` rows in the verify batch."""
    return t0 + c_row * rows


def _expected_tokens_tensor(
    widths: torch.Tensor, survival: torch.Tensor
) -> torch.Tensor:
    """Expected accepted tokens as a 0-dim tensor (no host sync)."""
    num_spec = survival.shape[1]
    clamped = widths.clamp(min=0, max=num_spec)
    prefix = torch.cumsum(survival, dim=1)
    picked = prefix.gather(1, clamped.clamp(min=1).unsqueeze(1) - 1).squeeze(1)
    accepted = torch.where(clamped > 0, picked, torch.zeros_like(picked))
    return (1.0 + accepted).sum()


def expected_tokens(widths: torch.Tensor, survival: torch.Tensor) -> float:
    """Expected accepted tokens for the given widths.

    Every request contributes its own sampled token plus one token per accepted
    draft; accepted drafts are ``sum_{j <= width} S_j`` in expectation.
    """
    return float(_expected_tokens_tensor(widths, survival).item())


def solve_verify_widths(
    survival: torch.Tensor,
    t0: float,
    c_row: float,
    other_rows: int = 0,
    c_req: float = 0.0,
    ragged_penalty: float = 0.0,
    max_iters: int = DEFAULT_ITERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Width per request maximizing expected accepted tokens per millisecond.

    Two families of widths are evaluated and the better one wins:

    * the per-request fixed point of ``S_j > c_row * Theta``, which is the
      optimum when a row costs the same everywhere, and
    * every uniform width, from zero (no speculation) to the full draft depth.

    Both are needed because the measured step cost is *concave* in rows — the
    weight and expert traffic is amortized over the batch, so a row costs 0.43 ms
    around eight rows but 0.18 ms around twenty. Under a concave cost each regime
    justifies itself locally (few rows look dear, many rows look cheap), so a
    threshold rule alone would lock into whichever basin it started in. Comparing
    the two families by their own tokens/ms picks the global one instead.

    Args:
        survival: ``[num_reqs, num_spec]`` probabilities that the first ``j``
            drafts of a request are all accepted. Need not be monotone.
        t0: Fixed per-step cost in milliseconds.
        c_row: Marginal cost of one more verified position, in milliseconds.
        other_rows: Rows the step will run that are not draft rows, i.e. prefill
            chunk tokens in a mixed batch. They are a fixed overhead that earns
            no decode token, so they lower the batch's tokens/ms and with it the
            bar a marginal draft row has to clear.
        c_req: Per-request cost in ms, from the same online fit that produces
            ``c_row``. It does not change the marginal condition — it is constant
            in the widths — but it belongs in the batch's tokens/ms, which is the
            price a marginal row is compared against.
        ragged_penalty: Extra milliseconds a *step* pays when the requests do not
            all verify the same width, also from the online fit. Serving mixed
            query lengths costs more than serving one, so a ragged split has to
            earn back that penalty before it is worth choosing.
        max_iters: Fixed-point iterations; the map converges in a few.

    Returns:
        The chosen widths, and the break-even threshold ``theta = c_row * Theta``
        in tokens per row, so a caller can check ``S_j > theta`` directly. Both
        stay on the input's device: reading either costs a host synchronization,
        which must happen once, in the caller.

    """
    num_spec = survival.shape[1]
    num_reqs = survival.shape[0]
    if num_spec <= 0 or num_reqs == 0:
        return torch.zeros(num_reqs, dtype=torch.long), torch.tensor(float("nan"))

    prefix = torch.cumsum(survival, dim=1)

    def tokens_of(widths: torch.Tensor) -> torch.Tensor:
        """Expected tokens per candidate, ``[C]`` for a ``[C, R]`` width matrix."""
        clamped = widths.clamp(min=1, max=num_spec)
        picked = prefix.unsqueeze(0).expand(widths.shape[0], -1, -1)
        picked = picked.gather(2, (clamped - 1).unsqueeze(-1)).squeeze(-1)
        accepted = torch.where(widths > 0, picked, torch.zeros_like(picked))
        return (1.0 + accepted).sum(dim=1)

    def cost_of(widths: torch.Tensor) -> torch.Tensor:
        rows = widths.sum(dim=1) + num_reqs + other_rows
        ragged = (widths.max(dim=1).values != widths.min(dim=1).values).to(rows.dtype)
        return t0 + c_row * rows + c_req * num_reqs + ragged_penalty * ragged

    # Family 1: the per-request fixed point, iterated from the full width.
    fixed = torch.full((num_reqs,), num_spec, dtype=torch.long, device=survival.device)
    for _ in range(max(1, max_iters)):
        theta = (
            c_row * tokens_of(fixed.unsqueeze(0))[0] / cost_of(fixed.unsqueeze(0))[0]
        )
        fixed = (survival > theta).sum(dim=1).clamp(max=num_spec)

    # Family 2: every uniform width, 0 (no speculation) through the full depth.
    uniform = (
        torch.arange(0, num_spec + 1, dtype=torch.long, device=survival.device)
        .unsqueeze(1)
        .expand(-1, num_reqs)
    )

    candidates = torch.cat([fixed.unsqueeze(0), uniform], dim=0)
    cost = cost_of(candidates)
    theta_vec = c_row * tokens_of(candidates) / cost
    # Indexed with the argmax tensor itself: taking .item() here would
    # synchronize the device on every step.
    best = theta_vec.argmax()
    return candidates[best], theta_vec[best]
