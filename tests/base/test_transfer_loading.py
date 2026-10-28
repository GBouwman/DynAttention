"""Transfer/loading regression tests: skip weights survive loading + setup('fit'),
time constants come from the config, identity transfer from a tau=dt backbone, LR groups."""
import copy
import logging

import pytest
import torch

from dynvision.models import DyRCNNx8
from dynvision.utils.model_utils import load_weights_for_training

H = 64
BASE = dict(input_dims=(1, 3, H, H), n_classes=10, dt=5, skip=True, recurrence_type="none")


@pytest.fixture(scope="module")
def backbone():
    """A 1-step, tau = dt backbone with skip connections and recognisable skip weights."""
    logging.disable(logging.WARNING)
    torch.manual_seed(0)
    m = DyRCNNx8(n_timesteps=1, tau=5, **BASE)
    m._initialize_connections()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.startswith("addskip") and ".source." not in name:
                p.fill_(0.123)
    m.eval()
    x = torch.rand(2, 1, 3, H, H)
    with torch.no_grad():
        ref = m(x)
    return m.state_dict(), x, (ref[:, -1] if ref.dim() == 3 else ref)


def _skip_weights(model):
    return [p for n, p in model.named_parameters()
            if n.startswith("addskip") and ".source." not in n and n.endswith("weight")]


def test_skip_weights_survive_load_and_setup(backbone):
    sd, _, _ = backbone
    m = DyRCNNx8(n_timesteps=1, tau=5, **BASE)
    keys = load_weights_for_training(m, sd)
    assert not keys["unexpected"], keys["unexpected"]
    m.setup("fit")  # what trainer.fit calls before training
    w = _skip_weights(m)
    assert w and all(torch.allclose(p, torch.full_like(p, 0.123)) for p in w)


def test_identity_transfer_same_dynamics_is_exact(backbone):
    sd, x, ref = backbone
    m = DyRCNNx8(n_timesteps=1, tau=5, **BASE)
    load_weights_for_training(m, sd)
    m.setup("fit")
    m.eval()
    with torch.no_grad():
        out = m(x)
    out = out[:, -1] if out.dim() == 3 else out
    assert torch.equal(out, ref)


def test_time_constants_come_from_config_and_transfer_converges(backbone):
    sd, x, ref = backbone
    m = DyRCNNx8(n_timesteps=1, tau=10, **BASE)  # alpha = dt / tau = 0.5
    keys = load_weights_for_training(m, sd)
    assert "tau_V1" in keys["kept_tau"]
    assert m.tau_V1.item() == pytest.approx(10.0)
    assert m.tstep_V1.tau.item() == pytest.approx(10.0)
    m.setup("fit")
    m.eval()
    with torch.no_grad():
        out = m(x.repeat(1, 40, 1, 1, 1))
    first, last = (out[:, 0] - ref).abs().max(), (out[:, -1] - ref).abs().max()
    assert first > 1e-2          # alpha = 0.5 really changes the early steps
    assert last < 1e-5           # and the settled state is the backbone's


def test_lr_groups_do_not_mutate_config():
    cfg = {"regular": {"lr_factor": 1.0}, "recurrence": {"lr_factor": 0.2}, "feedback": {"lr_factor": 0.5}}
    m = DyRCNNx8(n_timesteps=1, tau=5, lr_parameter_groups=copy.deepcopy(cfg),
                 **{**BASE, "recurrence_type": "full"})
    m._initialize_connections()
    for _ in range(2):
        groups = {g["name"]: g for g in m._group_lr_parameters(1e-3)}
        assert groups["recurrence"]["params"], "recurrent weights must land in the 'recurrence' group"
        assert groups["recurrence"]["lr"] == pytest.approx(2e-4)
    assert m.lr_parameter_groups == cfg


def test_default_lr_group_key_matches_parameter_names():
    from dynvision.params import model_params
    src = open(model_params.__file__).read()
    assert '"recurrent": {"lr_factor"' not in src


def test_train_model_imports():
    import dynvision.runtime.train_model  # noqa: F401
