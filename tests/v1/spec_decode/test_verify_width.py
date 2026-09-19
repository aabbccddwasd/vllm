# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the verify-width cost model."""

import random

import torch

from vllm.v1.spec_decode.verify_width import (
    RowCostModel,
    _expected_tokens_tensor,
    expected_tokens,
    solve_verify_widths,
    step_cost_ms,
)


def reported_theta(
    widths: torch.Tensor, survival: torch.Tensor, other_rows: int = 0
) -> float:
    """Theta must be the break-even in tokens/row, i.e. c_row * Theta."""
    rows = int(widths.sum().item()) + widths.numel() + other_rows
    return (
        C_ROW
        * float(_expected_tokens_tensor(widths, survival))
        / step_cost_ms(rows, T0, C_ROW)
    )


# Throughput of one step under a given width, the quantity being maximized.
def throughput(num_reqs: int, width: int, survival: torch.Tensor) -> float:
    widths = torch.full((num_reqs,), width)
    return expected_tokens(widths, survival) / step_cost_ms(
        num_reqs * (1 + width), T0, C_ROW
    )


# Calibrated per-position survival of DeepSeek-V4.1-Flash DSpark drafts.
SURVIVAL = torch.tensor([[0.625, 0.384, 0.252, 0.163, 0.110]])

# t(N) = 9.39 + 0.592 N ms measured over a single-request width sweep.
T0 = 9.39
C_ROW = 0.592


def test_theta_is_the_break_even_threshold():
    """Regression: theta used to be reported as Theta (tokens/ms), off by c_row.

    The decision read ``S_j > c_row * theta``, so widths were unaffected, but the
    logged number was unreadable and hid a cost model that was wrong at load.
    """
    for num_reqs, other_rows in ((1, 0), (4, 0), (12, 64)):
        survival = SURVIVAL.expand(num_reqs, -1).contiguous()
        widths, theta = solve_verify_widths(survival, T0, C_ROW, other_rows)
        want = reported_theta(widths, survival, other_rows)
        assert abs(float(theta) / want - 1.0) < 1e-3, (float(theta), want)
        # And the widths really are the positions that clear that threshold.
        assert widths.tolist() == [
            int(x) for x in (survival > float(theta)).sum(dim=1).clamp(max=5).tolist()
        ]


def test_returns_the_argmax_width():
    """The solved width must be the one no other width beats."""
    for num_reqs in (1, 2, 4, 8, 12):
        survival = SURVIVAL.expand(num_reqs, -1).contiguous()
        widths, theta = solve_verify_widths(survival, T0, C_ROW)
        best = max(
            throughput(num_reqs, k, survival) for k in range(survival.shape[1] + 1)
        )
        # theta is in tokens/row, so compare the throughput it implies.
        # The solve runs in float32 on the device, hence the loose tolerance.
        assert float(theta) / C_ROW >= best - 1e-5
        assert throughput(num_reqs, int(widths[0].item()), survival) >= best - 1e-5


def test_width_shrinks_as_the_batch_grows():
    """Throughput saturates, so a marginal row stops paying for itself.

    At one request a row is the only way to add tokens and the fixed cost hides
    it; by a dozen requests the step is already saturated.
    """
    seen = [
        int(
            solve_verify_widths(SURVIVAL.expand(n, -1).contiguous(), T0, C_ROW)[0][
                0
            ].item()
        )
        for n in (1, 2, 4, 8, 12)
    ]
    assert seen == [4, 3, 2, 1, 1], seen
    assert seen == sorted(seen, reverse=True)


def test_widths_are_per_request():
    """An unconfident request is verified more shallowly than a confident one."""
    survival = torch.stack([SURVIVAL[0], SURVIVAL[0] * 0.15])
    widths, _ = solve_verify_widths(survival, T0, C_ROW)
    assert widths[0] > widths[1], widths


def test_a_cheaper_row_buys_a_deeper_width():
    """Halving the per-row cost must not shrink the width."""
    for num_reqs in (1, 4, 12):
        survival = SURVIVAL.expand(num_reqs, -1).contiguous()
        cheap, _ = solve_verify_widths(survival, T0, C_ROW / 2)
        dear, _ = solve_verify_widths(survival, T0, C_ROW * 2)
        assert int(cheap[0].item()) >= int(dear[0].item())


def test_flat_survival_keeps_every_position():
    """If every draft is certain, every row is free money."""
    widths, _ = solve_verify_widths(torch.ones(4, 5), T0, C_ROW)
    assert widths.tolist() == [5, 5, 5, 5]


def test_prefill_rows_are_priced_in():
    """A step already carrying a prefill chunk loses less by verifying deeper.

    Prefill rows are a fixed overhead that earns no decode token, so they lower
    the batch's tokens/ms and with it the bar a marginal draft row must clear.
    """
    survival = SURVIVAL.expand(4, -1).contiguous()
    widths = [
        int(solve_verify_widths(survival, T0, C_ROW, o)[0][0]) for o in (0, 8, 64, 512)
    ]
    assert widths == sorted(widths)
    assert widths[0] == 2 and widths[-1] == 5


def test_a_very_large_batch_drops_speculation_entirely():
    """Past ~32 requests one more row costs more than it accepts.

    Width 0 is a legal answer, not a bug: the scheduler already handles a
    request whose draft list is empty (it is the plain decode path), and this
    configuration never reaches it because max_num_seqs is 12.
    """
    survival = SURVIVAL.expand(64, -1).contiguous()
    widths, _ = solve_verify_widths(survival, T0, C_ROW)
    assert widths.tolist() == [0] * 64


# --- online cost model -------------------------------------------------------


def test_online_model_recovers_the_true_curve():
    """The engine's own step samples must identify the cost it is running at."""
    model = RowCostModel(t0=T0, c_row=C_ROW)
    rng = random.Random(0)
    for _ in range(200):
        num_reqs = rng.choice([1, 2, 4, 8])
        rows = rng.choice([8, 12, 16, 24, 32])
        true_ms = 12.0 + 0.15 * rows + 1.5 * num_reqs
        model.observe(rows, num_reqs, true_ms * rng.uniform(0.98, 1.02))
    t0, c_row, c_req, _ = model.estimate()
    # The three terms are only individually identifiable if the regressors are
    # spread; what must hold is the row price, which is what the decision spends.
    assert abs(c_row - 0.15) < 0.03
    for num_reqs, rows in ((1, 12), (4, 24), (8, 48)):
        predicted = t0 + c_row * rows + c_req * num_reqs
        assert abs(predicted - (12.0 + 0.15 * rows + 1.5 * num_reqs)) < 1.0


def test_online_model_withholds_a_fit_it_cannot_identify():
    """One batch size identifies an intercept, not a slope."""
    model = RowCostModel(t0=T0, c_row=C_ROW)
    for _ in range(200):
        model.observe(24, 4, 24.0)
    assert model.estimate() is None
    for _ in range(200):
        model.observe(24, 4, 24.0)
    assert model.estimate() is None


def test_online_model_needs_enough_samples():
    model = RowCostModel(t0=T0, c_row=C_ROW, min_samples=8)
    for rows in (8, 32):
        model.observe(rows, 4, 9.0 + 0.6 * rows)
    assert model.estimate() is None


def test_online_model_tracks_a_load_change():
    """Forgetting is what lets a cheaper row regime take over."""
    model = RowCostModel(t0=T0, c_row=C_ROW, forgetting=0.9)
    rng = random.Random(1)
    for _ in range(50):
        rows = rng.choice([8, 16, 24])
        model.observe(rows, 4, 9.4 + 0.592 * rows)
    assert abs(model.estimate()[1] - 0.592) < 0.05
    for _ in range(120):
        rows = rng.choice([8, 16, 24])
        model.observe(rows, 4, 14.0 + 0.02 * rows)
    assert model.estimate()[1] < 0.1


def test_online_model_ignores_garbage_samples():
    model = RowCostModel(t0=T0, c_row=C_ROW)
    for _ in range(50):
        model.observe(0, 4, 10.0)
        model.observe(12, 4, 0.0)
        model.observe(12, 4, float("nan"))
        model.observe(12, 0, 10.0)
    assert model.num_samples == 0
    assert model.estimate() is None


def test_online_model_rejects_an_absurd_slope():
    """A negative per-row cost is unphysical; better to keep the prior."""
    model = RowCostModel(t0=T0, c_row=C_ROW)
    for rows in (8, 32) * 20:
        model.observe(rows, 4, 30.0 - 0.4 * rows)
    assert model.estimate() is None


def test_online_model_separates_rows_from_requests():
    """Rows and requests must not be folded together.

    With a fixed width the two move together; the per-request regressor is what
    keeps the per-request costs out of the row price. Here the requests carry a
    large cost of their own, so a two-parameter fit would inflate c_row to
    roughly (3.0 * 6) / 6 = 3 ms per row.
    """
    model = RowCostModel(t0=T0, c_row=C_ROW, forgetting=1.0)
    rng = random.Random(2)
    for _ in range(400):
        num_reqs = rng.choice([1, 2, 4, 8])
        width = rng.choice([1, 3, 5])
        rows = num_reqs * (1 + width)
        model.observe(rows, num_reqs, 6.0 + 0.15 * rows + 3.0 * num_reqs)
    t0, c_row, c_req, _ = model.estimate()
    assert abs(c_row - 0.15) < 0.05, c_row
    assert abs(c_req - 3.0) < 0.5, c_req
    assert abs(t0 - 6.0) < 1.0, t0


# --- global choice between a ragged split and a uniform width -----------------


def test_ragged_split_must_earn_back_its_penalty():
    """Serving mixed query lengths costs more, so the split has to pay for it.

    One confident and one unconfident request would be split 5/1 without a
    penalty; with a penalty large enough to swamp the extra tokens the solver
    must fall back to verifying the same width for both.
    """
    survival = torch.stack([SURVIVAL[0], SURVIVAL[0] * 0.15])
    plain, _ = solve_verify_widths(survival, T0, C_ROW, ragged_penalty=0.0)
    assert plain[0] > plain[1], plain  # a genuine split
    penalised, _ = solve_verify_widths(survival, T0, C_ROW, ragged_penalty=5.0)
    assert penalised[0] == penalised[1], penalised


def test_solver_picks_the_best_candidate_not_the_first():
    """The answer must be the global argmax over both width families."""
    survival = SURVIVAL.expand(4, -1).contiguous()
    others = (0.0, 2.0)
    for penalty in others:
        widths, theta = solve_verify_widths(survival, T0, C_ROW, ragged_penalty=penalty)
        candidates = [torch.full((4,), k) for k in range(6)] + [widths]
        best = max(
            expected_tokens(c, survival) / step_cost_ms(int(c.sum()) + 4, T0, C_ROW)
            for c in candidates
            if c.max() == c.min() or c is widths
        )
        assert float(theta) / C_ROW >= best - 1e-5


def test_online_model_learns_the_ragged_penalty():
    """Mixed verify widths cost extra per step, and the fit must attribute it.

    Without the ragged term this penalty would land on c_row and make shallow
    splits look cheaper than they are.
    """
    model = RowCostModel(t0=T0, c_row=C_ROW, forgetting=1.0)
    rng = random.Random(3)
    for _ in range(400):
        num_reqs = rng.choice([2, 4])
        width = rng.choice([1, 3, 5])
        rows = num_reqs * (1 + width)
        ragged = rng.random() < 0.5
        model.observe(
            rows,
            num_reqs,
            6.0 + 0.2 * rows + 2.0 * num_reqs + (2.5 if ragged else 0.0),
            ragged=ragged,
        )
    t0, c_row, c_req, c_ragged = model.estimate()
    assert abs(c_row - 0.2) < 0.05, c_row
    assert abs(c_ragged - 2.5) < 0.6, c_ragged


def test_only_the_cost_ratio_matters():
    """The decision is homogeneous in the cost model.

    A deployment supplies one number — the per-token verify cost — and the
    engine measures the rest, so what the solver must depend on is the *ratio*
    between the row price and the batch costs, never their absolute scale.
    """
    survival = SURVIVAL.expand(4, -1).contiguous()
    base = solve_verify_widths(survival, 13.6, C_ROW, 0, 2.0, 1.5)
    for scale in (0.25, 4.0, 100.0):
        scaled = solve_verify_widths(
            survival, 13.6 * scale, C_ROW * scale, 0, 2.0 * scale, 1.5 * scale
        )
        assert scaled[0].tolist() == base[0].tolist()


def test_a_cheaper_token_buys_a_deeper_width_than_a_dearer_one():
    """The per-token cost is the knob, and it must move the width monotonically."""
    survival = SURVIVAL.expand(4, -1).contiguous()
    widths = [
        int(solve_verify_widths(survival, 13.6, c, 0, 2.0)[0][0])
        for c in (0.05, 0.2, 0.6, 1.5)
    ]
    assert widths == sorted(widths, reverse=True), widths
    assert widths[0] == 5 and widths[-1] <= 2


def test_every_declared_env_has_a_resolver():
    """A declared-but-unresolvable env raises AttributeError in the engine.

    Regression guard: consolidating the decision knobs once deleted the resolver
    entries for the confidence-head switches while leaving their declarations, and
    every engine start then died with AttributeError at import time.
    """
    import ast
    import pathlib

    import vllm.envs as envs_module

    # The declarations live under `if TYPE_CHECKING`, so read the source rather
    # than the runtime namespace, which would make this guard vacuous.
    tree = ast.parse(pathlib.Path(envs_module.__file__).read_text())
    declared = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("VLLM_")
    }
    assert len(declared) > 200, len(declared)
    resolvable = set(envs_module.environment_variables)
    assert not (declared - resolvable), sorted(declared - resolvable)


class _FakeInputBatch:
    """Just the fields the width decision reads."""

    def __init__(self, num_reqs: int = 4, draft_counts=(5, 5, 3, 5)):
        import numpy as np

        self.num_reqs = num_reqs
        self.num_tokens = sum(1 + c for c in draft_counts)
        self.num_draft_tokens = sum(draft_counts)
        self.num_tokens_after_padding = self.num_tokens
        self.num_draft_tokens_per_req = np.array(draft_counts)
        self.has_prefill = False


def test_decision_is_silent_until_the_cost_curve_is_measured():
    """Regression: the cold start must not crash the step.

    The engine reports no width until its own fit is identified, and the callers
    used to unpack that None unconditionally ("cannot unpack non-iterable
    NoneType object" killed every engine start).
    """
    from vllm.v1.spec_decode.verify_width import RowCostModel
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.row_cost_model = RowCostModel(t0=0.0, c_row=C_ROW)
    runner.num_speculative_steps = 5

    survival = SURVIVAL.expand(4, -1).contiguous()
    assert runner._spec_widths_from_survival(survival) is None
    assert runner._row_cost() is None

    # Once identified, the same call returns widths instead.
    for rows in (8, 24):
        for _ in range(20):
            runner.row_cost_model.observe(rows, 4, 11.0 + 0.2 * rows)
    result = runner._spec_widths_from_survival(survival)
    assert result is not None
    widths, _ = result
    assert widths.shape == (4,)


def test_the_specified_token_cost_carries_the_decision_before_the_fit():
    """One batch size identifies the intercept, not the slope.

    Until the engine has seen two sizes, the deployment's per-token verify cost
    is the row price and the measured intercept fills the rest, so the policy
    engages after a single window instead of waiting for row spread.
    """
    from vllm.v1.spec_decode.verify_width import RowCostModel
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.row_cost_model = RowCostModel(t0=0.0, c_row=0.592)
    runner.num_speculative_steps = 5
    for _ in range(20):  # one size only: no row spread
        runner.row_cost_model.observe(24, 4, 19.0)
    assert runner.row_cost_model.estimate() is None
    cost = runner._row_cost()
    assert cost is not None
    t0, c_row, c_req, c_ragged = cost
    assert c_row == 0.592  # the specified hyperparameter, not a fitted value
    assert t0 > 0.0 and c_req >= 0.0 and c_ragged == 0.0
    result = runner._spec_widths_from_survival(SURVIVAL.expand(4, -1).contiguous())
    assert result is not None and result[0].shape == (4,)


def test_probe_cost_fields_survives_a_missing_cost_model():
    """The diagnostic probe must not depend on solver internals.

    It used to print `t0/c_row/c_req`, which only exist inside
    `_spec_widths_from_survival`, so enabling the probe crashed the worker with
    `name 't0' is not defined` on the first sampled step (observed 2026-09-15).
    """
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    class Stub:
        pass

    stub = Stub()
    stub._row_cost = lambda: (4.1, 0.43, 0.12, 0.02)
    source, t0, c_row, c_req = GPUModelRunner._probe_cost_fields(stub)
    assert (source, t0, c_row, c_req) == ("online", 4.1, 0.43, 0.12)

    stub._row_cost = lambda: None
    source, t0, c_row, c_req = GPUModelRunner._probe_cost_fields(stub)
    assert source == "unknown"
    assert t0 != t0 and c_row != c_row and c_req != c_req  # NaN


def test_width_probe_prints_only_defined_names():
    """Structural guard: the probe call sites must not name solver locals."""
    import ast
    from pathlib import Path

    src = Path("vllm/v1/worker/gpu/model_runner.py").read_text()
    tree = ast.parse(src)
    leaked = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "info"):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.Name) and arg.id in {"t0", "c_row", "c_req"}:
                leaked.append(f"line {arg.lineno}: {arg.id}")
    assert not leaked, f"probe logs solver-internal names: {leaked}"
