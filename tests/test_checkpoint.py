"""Checkpoint save/load consistency, exact resumption and deterministic L-BFGS closure."""
import shutil

import pytest
import torch

from cavity_pinn.config import load_config
from cavity_pinn.train import Trainer

TINY = ["model.width=8", "model.depth=2", "sampling.n_interior=64", "sampling.n_boundary_per_wall=16",
        "sampling.refresh_every=3", "adam.iterations=8", "adam.decay_steps=4", "lbfgs.iterations=12",
        "lbfgs.max_iter_per_step=4", "training.log_every=2", "training.checkpoint_every=2",
        "training.keep_checkpoints=true", "training.num_threads=1"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    cfg = load_config("config/re100.yaml", TINY)
    d = tmp_path_factory.mktemp("full")
    tr = Trainer(cfg, 101, d, verbose=False)
    tr.run()
    return tr, d


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_checkpoint_roundtrip(full_run):
    tr, d = full_run
    re = Trainer.from_checkpoint(d / "checkpoints" / "checkpoint_final.pt", verbose=False)
    assert torch.equal(_params(re.model), _params(tr.model))
    x = torch.rand(20, 2)
    assert torch.equal(re.model(x), tr.model(x))
    assert torch.equal(re.X, tr.X) and re.phase == tr.phase and re.n_evals == tr.n_evals
    assert re.cfg == tr.cfg and len(re.history) == len(tr.history)


@pytest.mark.parametrize("snap", ["snapshot_adam_000004.pt", "snapshot_lbfgs_000008.pt"])
def test_resume_reproduces_uninterrupted_run(full_run, tmp_path, snap):
    tr, d = full_run
    (tmp_path / "checkpoints").mkdir()
    shutil.copy(d / "checkpoints" / snap, tmp_path / "checkpoints" / snap)
    re = Trainer.from_checkpoint(tmp_path / "checkpoints" / snap, verbose=False)
    re.run()
    assert re.adam_it == tr.adam_it and re.lbfgs_it == tr.lbfgs_it and re.n_evals == tr.n_evals
    torch.testing.assert_close(_params(re.model), _params(tr.model), rtol=0, atol=0)


def test_lbfgs_closure_is_deterministic(full_run):
    tr, _ = full_run
    out = []
    for _ in range(2):
        tr.model.zero_grad()
        total, _ = tr.loss()
        total.backward()
        out.append((total.item(), torch.cat([p.grad.flatten() for p in tr.model.parameters()])))
    assert out[0][0] == out[1][0] and torch.equal(out[0][1], out[1][1])


def test_history_logs_unweighted_components_and_weights(full_run):
    tr, _ = full_run
    h = tr.history[-1]
    for k in ("continuity", "momentum_u", "momentum_v", "wall_top", "wall_bottom", "wall_left",
              "wall_right", "gauge", "w_wall", "n_evals", "elapsed_s", "peak_mem_mb", "lr"):
        assert k in h
    w = tr.weights
    recomposed = (w["continuity"] * h["continuity"] + w["momentum_u"] * h["momentum_u"]
                  + w["momentum_v"] * h["momentum_v"] + w["gauge"] * h["gauge"]
                  + w["wall"] * sum(h[f"wall_{x}"] for x in ("top", "bottom", "left", "right")))
    assert recomposed == pytest.approx(h["total"], rel=1e-12)
    assert {r["phase"] for r in tr.history} == {"adam", "lbfgs"}
