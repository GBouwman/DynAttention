"""Topography on the channel sheet: does feature tuning vary smoothly across a sqrt(C) x sqrt(C) sheet?

Two signals are measured per area:
- **drive**: the area's feedforward conv output, before any sheet pooling or recurrence. Its
  topography comes only from the learned weights, so this is the real test of map formation.
- **output**: what the area passes on. For B-QG this is smoothed by construction (TI > 0 even at
  random initialization), and for Rq partly so; report it, but judge maps by the drive.

For every area of each model, channels are probed with full-field gratings (8 orientations x
3 spatial frequencies x 4 phases) and 12 hue patches. From the tuning matrix it reports

- orientation preference and selectivity per channel, drawn as a map on the sheet;
- tuning correlation between channel pairs as a function of their torus distance on the sheet;
- a topography index TI = mean correlation of sheet neighbours (d <= 1.5) minus that of all pairs,
  with a z-score against shuffled sheet positions.

A plain backbone is the control: its channel order is arbitrary, so TI ~ 0.

Usage:
    python -m dynvision.runtime.analyze_topography \\
        --model .../DyRCNNx8:...+sheetpool=true_7000/miniecoset/trained.pt --label B-QG \\
        --model .../DyRCNNx8:..._7000/miniecoset/trained.pt --label plain \\
        --out topography
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from dynvision.utils.model_utils import load_trained_model

N_ORI, SFS, N_PHASE, N_HUE = 8, (4, 8, 16), 4, 12


def make_stimuli(res: int) -> tuple[torch.Tensor, dict]:
    """Gratings (grey, contrast 0.5) then hue patches, in [0, 1]. Returns [N, 3, res, res] and an index."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
    stims, ori_index = [], []
    for o in range(N_ORI):
        th = math.pi * o / N_ORI
        for sf in SFS:
            for p in range(N_PHASE):
                g = 0.5 + 0.25 * torch.sin(2 * math.pi * sf * (xx * math.cos(th) + yy * math.sin(th)) + 2 * math.pi * p / N_PHASE)
                stims.append(g.expand(3, res, res))
                ori_index.append(o)
    for h in range(N_HUE):
        hue = h / N_HUE
        rgb = [0.5 + 0.25 * math.cos(2 * math.pi * (hue - k / 3)) for k in range(3)]
        stims.append(torch.tensor(rgb)[:, None, None].expand(3, res, res))
    return torch.stack(stims), {"ori": np.array(ori_index), "n_grating": len(ori_index)}


def _centre_mean(h: torch.Tensor) -> torch.Tensor:
    H, W = h.shape[-2:]
    return h.float()[..., H // 4: H - H // 4 or None, W // 4: W - W // 4 or None].mean(dim=(-2, -1)).cpu()


@torch.no_grad()
def area_responses(model, stimuli: torch.Tensor, mean, std, n_timesteps: int, device, batch: int = 16) -> dict:
    """Spatial mean over the central half of each area's drive and output, after the last timestep."""
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    last_drive, hooks = {}, []
    for name in model.layer_names:
        layer = getattr(model, name)
        conv = layer.conv2 if getattr(layer, "mid_channels", None) else layer.conv   # the area's feedforward output conv
        hooks.append(conv.register_forward_hook(lambda m, i, o, name=name: last_drive.__setitem__(name, o)))
    out = {kind: {name: [] for name in model.layer_names} for kind in ("drive", "output")}
    try:
        for i in range(0, len(stimuli), batch):
            x = ((stimuli[i:i + batch].to(device) - mean) / std)
            model(x[:, None].repeat(1, n_timesteps, 1, 1, 1))
            for name in model.layer_names:
                out["drive"][name].append(_centre_mean(last_drive[name]))
                out["output"][name].append(_centre_mean(getattr(model, name).get_newest_hidden_state()))
    finally:
        for h in hooks:
            h.remove()
    return {kind: {name: torch.cat(v).numpy() for name, v in d.items()} for kind, d in out.items()}


def torus_distance(side: int) -> np.ndarray:
    pos = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2)
    d = np.abs(pos[:, None] - pos[None])
    d = np.minimum(d, side - d)
    return np.sqrt((d ** 2).sum(-1))


def topography(resp: np.ndarray, index: dict, rng: np.random.Generator, n_shuffle: int = 200) -> dict:
    n_c = resp.shape[1]
    side = int(round(math.sqrt(n_c)))
    if side * side != n_c:
        return {"skipped": f"{n_c} channels is not a square"}
    alive = resp.std(0) > 1e-8
    z = (resp - resp.mean(0)) / np.where(alive, resp.std(0), 1.0)
    corr = (z.T @ z) / len(z)
    dist = torus_distance(side)
    off = ~np.eye(n_c, dtype=bool) & alive[:, None] & alive[None]
    near = off & (dist <= 1.5)

    def ti(perm):
        c = corr[np.ix_(perm, perm)]
        return c[near].mean() - c[off].mean()

    ident = np.arange(n_c)
    t0 = ti(ident)
    null = np.array([ti(rng.permutation(n_c)) for _ in range(n_shuffle)])
    bins = np.arange(0.5, dist.max() + 1.0, 1.0)
    curve = [float(corr[off & (dist > lo) & (dist <= lo + 1)].mean()) if (off & (dist > lo) & (dist <= lo + 1)).any() else float("nan")
             for lo in np.concatenate([[0.0], bins[:-1]])]

    g = resp[: index["n_grating"]]
    r_ori = np.stack([g[index["ori"] == o].mean(0) for o in range(N_ORI)])          # [N_ORI, C]
    r_ori = np.clip(r_ori - r_ori.min(0, keepdims=True), 0, None)
    vec = (r_ori * np.exp(2j * np.pi * np.arange(N_ORI) / N_ORI)[:, None]).sum(0)
    osi = np.abs(vec) / np.maximum(r_ori.sum(0), 1e-12)
    pref = (np.angle(vec) % (2 * np.pi)) / 2                                         # radians, 0..pi
    return {
        "side": side,
        "TI": float(t0),
        "TI_z": float((t0 - null.mean()) / (null.std() + 1e-12)),
        "alive_fraction": float(alive.mean()),
        "corr_vs_distance": curve,
        "orientation_pref": pref.reshape(side, side).tolist(),
        "orientation_selectivity": osi.reshape(side, side).tolist(),
    }


def plot(results: dict, path: Path, kind: str = "drive") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    labels = list(results)
    areas = list(results[labels[0]][kind])
    fig, axes = plt.subplots(len(labels), len(areas) + 1, figsize=(3.1 * (len(areas) + 1), 3.0 * len(labels)), squeeze=False)
    for i, lab in enumerate(labels):
        for j, area in enumerate(areas):
            ax = axes[i, j]
            r = results[lab][kind][area]
            if "skipped" in r:
                ax.axis("off"); continue
            pref = np.array(r["orientation_pref"]); osi = np.array(r["orientation_selectivity"])
            val = np.clip(osi / max(osi.max(), 1e-9), 0.15, 1.0)
            ax.imshow(hsv_to_rgb(np.stack([pref / np.pi, np.ones_like(pref), val], -1)), interpolation="nearest")
            ax.set_title(f"{lab} {area} ({kind})\nTI {r['TI']:.2f} (z {r['TI_z']:.0f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[i, -1]
        for k_, area in enumerate(areas):
            for kd, ls in (("drive", "-"), ("output", ":")):
                r = results[lab][kd][area]
                if "skipped" not in r:
                    ax.plot(np.arange(len(r["corr_vs_distance"])), r["corr_vs_distance"], ls, color=f"C{k_}", marker="o", ms=3,
                            label=f"{area} {kd}" if kd == "drive" else None)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("sheet distance (torus)"); ax.set_ylabel("tuning correlation"); ax.legend(fontsize=7)
        ax.set_title(f"{lab}: corr vs distance (solid drive, dotted output)", fontsize=8)
    fig.text(0.01, 0.005, "Maps: hue = preferred orientation, brightness = selectivity", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="trained.pt (with trained.pt.config.yaml next to it)")
    ap.add_argument("--label", action="append", help="one label per --model")
    ap.add_argument("--out", default="topography", help="output prefix (writes <out>.json and <out>.png)")
    ap.add_argument("--timesteps", type=int, default=None, help="override presentation length (default: the model's n_timesteps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    labels = args.label or [Path(m).parent.parent.name for m in args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    results = {}
    for path, lab in zip(args.model, labels):
        model, cfg = load_trained_model(path, device=device)
        mean, std = cfg["data.normalize"]
        T = args.timesteps or int(getattr(model, "n_timesteps", 1))
        stimuli, index = make_stimuli(int(cfg["data.resolution"]))
        resp = area_responses(model, stimuli, mean, std, T, device)
        results[lab] = {kind: {area: topography(r, index, rng) for area, r in d.items()} for kind, d in resp.items()}
        for kind in ("drive", "output"):
            print(f"{lab} {kind}: TI (z) per area", {a: (round(v["TI"], 3), round(v["TI_z"], 1)) for a, v in results[lab][kind].items() if "TI" in v})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    plot(results, out.with_suffix(".png"))
    print("wrote", out.with_suffix(".json"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
