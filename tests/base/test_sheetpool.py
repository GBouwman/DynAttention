"""SheetPool (Qian et al. 2026 topographic pooling) and its DyRCNN integration."""
import logging
import math

import pytest
import torch

from dynvision.model_components import SheetPool
from dynvision.models import DyRCNNx8
from dynvision.utils.model_utils import load_weights_for_training

logging.disable(logging.WARNING)
H = 64
BASE = dict(input_dims=(1, 3, H, H), n_classes=10, dt=5, tau=5, n_timesteps=1, skip=True, recurrence_type="none")


def test_kernel_is_a_torus_gaussian_average():
    pool = SheetPool(64, sigma_floor=0.5)
    pool.set_sigma(1.0)
    k = pool.kernel()
    assert torch.allclose(k.sum(1), torch.ones(64))
    assert torch.allclose(k, k.T, atol=1e-6)                       # symmetric on the torus
    # channel 0 = sheet (0,0); its neighbours (0,1) and (0,7) are equally close on the torus
    assert torch.isclose(k[0, 1], k[0, 7]) and torch.isclose(k[0, 8], k[0, 56])
    assert k[0, 0] > k[0, 1] > k[0, 2]


def test_sigma_floor_and_non_square():
    pool = SheetPool(144, sigma_floor=0.75)
    pool.set_sigma(0.01)
    assert pool.sigma.item() == pytest.approx(0.75)
    with pytest.raises(ValueError):
        SheetPool(100 + 1)


def test_tiny_sigma_is_the_plain_network():
    torch.manual_seed(0)
    plain = DyRCNNx8(**BASE)
    plain._initialize_connections()
    pooled = DyRCNNx8(sheetpool=True, sheetpool_sigma_floor=0.0, **BASE)
    load_weights_for_training(pooled, plain.state_dict())
    for name in pooled.layer_names:
        getattr(pooled, f"sheetpool_{name}").set_sigma(1e-3)       # K = identity
    x = torch.rand(2, 1, 3, H, H)
    plain.eval(); pooled.eval()
    with torch.no_grad():
        assert torch.allclose(plain(x), pooled(x), atol=1e-6)
        for name in pooled.layer_names:
            getattr(pooled, f"sheetpool_{name}").set_sigma(2.0)
        assert not torch.allclose(plain(x), pooled(x), atol=1e-3)  # pooling really acts


def test_operation_order_and_schedule():
    m = DyRCNNx8(sheetpool=True, **BASE)
    ops = m.layer_operations
    assert ops.index("nonlin") < ops.index("sheetpool") < ops.index("record") < ops.index("delay")
    m._update_sheetpool_sigma(0)
    assert m.sheetpool_V1.sigma.item() == pytest.approx(4.0)        # half of the 8 x 8 sheet
    assert m.sheetpool_IT.sigma.item() == pytest.approx(11.5)       # half of the 23 x 23 sheet
    m._update_sheetpool_sigma(20)
    assert m.sheetpool_IT.sigma.item() == pytest.approx(11.5 * math.exp(-20 / 22.5))
    m._update_sheetpool_sigma(200)
    assert m.sheetpool_V1.sigma.item() == pytest.approx(0.75)       # floor


def test_trained_sigma_is_saved_and_restored():
    m = DyRCNNx8(sheetpool=True, **BASE)
    m._update_sheetpool_sigma(60)
    sd = m.state_dict()
    assert "sheetpool_V4.sigma" in sd
    stage2 = DyRCNNx8(sheetpool=True, sheetpool_schedule=False, **{**BASE, "tau": 10})
    load_weights_for_training(stage2, sd)
    assert stage2.sheetpool_V4.sigma.item() == pytest.approx(m.sheetpool_V4.sigma.item())


def test_gradients_flow_through_the_pool():
    m = DyRCNNx8(sheetpool=True, **BASE)
    m._initialize_connections()
    out = m(torch.rand(2, 1, 3, H, H))
    out.sum().backward()
    g = m.V1.conv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_plain_model_unchanged():
    m = DyRCNNx8(**BASE)
    assert "sheetpool" not in m.layer_operations
    assert not any(k.startswith("sheetpool") for k in m.state_dict())
