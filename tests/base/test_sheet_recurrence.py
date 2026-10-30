"""Rq: fixed sheet-Gaussian recurrence (rctype=sheet)."""
import logging
import math

import pytest
import torch

from dynvision.model_components.sheet_recurrence import SheetRecurrence
from dynvision.models import DyRCNNx8

logging.disable(logging.WARNING)
H = 64
RQ = dict(input_dims=(1, 3, H, H), n_classes=10, dt=5, tau=10, t_recurrence=5, skip=True, recurrence_type="sheet")


def test_module_is_gain_times_sheet_average():
    rec = SheetRecurrence(64)
    x = torch.rand(2, 64, 5, 5)
    y = rec(x)
    assert y.shape == x.shape
    assert torch.allclose(y.sum(1), 0.3 * x.sum(1), atol=1e-5)    # rows of K sum to 1, gain 0.3
    assert not any(p.requires_grad for p in rec.parameters())       # nothing is learned


def _late_change(rctype):
    torch.manual_seed(0)
    m = DyRCNNx8(n_timesteps=16, **{**RQ, "recurrence_type": rctype})
    m._initialize_connections()
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(2, 30, 3, H, H))
    assert torch.isfinite(out).all()
    return m, max(((out[:, t] - out[:, t - 1]).abs().max() / out[:, t].abs().max()).item() for t in range(20, 30))


def test_rq_model_builds_and_is_as_stable_as_no_recurrence():
    m, rq = _late_change("sheet")
    assert isinstance(m.V1.recurrence, SheetRecurrence) and isinstance(m.IT.recurrence, SheetRecurrence)
    _, none = _late_change("none")
    assert rq <= 1.5 * none + 1e-6   # a loop gain of 0.3 adds no instability


def test_schedule_reaches_the_recurrent_pools():
    m = DyRCNNx8(n_timesteps=4, **RQ)
    m._update_sheetpool_sigma(0)
    assert m.V1.recurrence.pool.sigma.item() == pytest.approx(4.0)
    m._update_sheetpool_sigma(20)
    assert m.IT.recurrence.pool.sigma.item() == pytest.approx(11.5 * math.exp(-20 / 22.5))
    assert "V2.recurrence.pool.sigma" in m.state_dict()


def test_gradients_reach_feedforward_weights():
    m = DyRCNNx8(n_timesteps=4, **RQ)
    m._initialize_connections()
    m(torch.rand(2, 4, 3, H, H)).sum().backward()
    g = m.V1.conv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
