"""Qian-style topography figures for DynVision models (feature maps, category maps, time course).

For each model and area, units are laid out on their sqrt(C) x sqrt(C) channel sheet and coloured by
- preferred orientation and spatial frequency (full-field gratings),
- chromaticity preference: t-value of colour vs grey-scale versions of the same images,
- animate vs inanimate preference (t-value, mini-ecoset val images),
- human (man / woman / child) vs everything else (t-value; a face/body-patch proxy).

It also plots tuning correlation against sheet distance for natural images, with an exponential fit
(Qian et al. 2026 Fig. 2b style), and, for recurrent models, the topography index per timestep.

Maps are computed from the *drive* (an area's feedforward conv output, before sheet pooling or
recurrence), which only reflects learned weights. Use --signal output for what the area passes on.

Usage:
    python -m dynvision.runtime.topography_maps --h5 <mini-ecoset .h5> \\
        --model <trained.pt> --label B-QG --model <trained.pt> --label plain --out <dir>
"""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dynvision.runtime.analyze_topography import make_stimuli, topography, torus_distance
from dynvision.utils.model_utils import load_trained_model

ANIMATE = {
    "man", "woman", "child", "monkey", "elephant", "dog", "pig", "cat", "tiger", "leopard", "zebra", "wolf",
    "camel", "kangaroo", "deer", "rhino", "cow", "goat", "mouse", "squirrel", "bird", "snake", "turtle", "frog",
    "spider", "rabbit", "alligator", "cockroach", "chicken", "wasp", "butterfly", "snail", "jellyfish", "worm",
    "grasshopper", "lizard", "mosquito", "lobster", "ant", "beetle", "bear", "horse", "lion",
}
HUMAN = {"man", "woman", "child"}


# ------------------------------------------------------------------ stimuli
def load_val_images(h5_path: str, per_class: int, res: int, seed: int = 0):
    """per_class val images of every category, resized + centre-cropped to res, in [0, 1]."""
    with h5py.File(h5_path, "r") as f:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in f["categories"][:]]
        labels = f["val"]["labels"][:]
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == k), per_class, replace=False)
                                      for k in range(len(cats))]))
        imgs = torch.from_numpy(f["val"]["data"][idx]).permute(0, 3, 1, 2).float() / 255.0
    side = min(imgs.shape[-2:])
    imgs = F.interpolate(imgs, scale_factor=res / side, mode="bilinear", antialias=True, align_corners=False)
    H, W = imgs.shape[-2:]
    top, left = (H - res) // 2, (W - res) // 2
    imgs = imgs[..., top:top + res, left:left + res].contiguous()
    names = [cats[k] for k in labels[idx]]
    return imgs, names


def greyscale(imgs: torch.Tensor) -> torch.Tensor:
    lum = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
    return lum[:, None].expand(-1, 3, -1, -1).contiguous()


# ------------------------------------------------------------------ responses at every timestep
@torch.no_grad()
def responses_over_time(model, stimuli, mean, std, n_timesteps, device, batch=16):
    """Returns {"drive"|"output": {area: [T, N, C]}}; spatial mean over the whole map."""
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    steps = {"drive": {}, "output": {}}
    hooks = []
    for name in model.layer_names:
        layer = getattr(model, name)
        conv = layer.conv2 if getattr(layer, "mid_channels", None) else layer.conv
        hooks.append(conv.register_forward_hook(
            lambda m, i, o, name=name: steps["drive"].setdefault(name, []).append(o.float().mean((-2, -1)).cpu())))
        pool = getattr(model, f"sheetpool_{name}", None)
        tstep = getattr(model, f"tstep_{name}")

        def out_hook(m, i, o, name=name, pool=pool):
            y = F.relu(o)
            if pool is not None:
                y = pool(y)
            steps["output"].setdefault(name, []).append(y.float().mean((-2, -1)).cpu())
        hooks.append(tstep.register_forward_hook(out_hook))

    result = {"drive": {n: [] for n in model.layer_names}, "output": {n: [] for n in model.layer_names}}
    try:
        for i in range(0, len(stimuli), batch):
            for d in steps.values():
                d.clear()
            x = (stimuli[i:i + batch].to(device) - mean) / std
            model(x[:, None].repeat(1, n_timesteps, 1, 1, 1))
            for kind in result:
                for name in model.layer_names:
                    result[kind][name].append(torch.stack(steps[kind][name][-n_timesteps:]))   # [T, B, C]
    finally:
        for h in hooks:
            h.remove()
    return {k: {n: torch.cat(v, dim=1).numpy() for n, v in d.items()} for k, d in result.items()}


# ------------------------------------------------------------------ statistics
def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb + 1e-12)


def corr_vs_distance(resp: np.ndarray):
    n_c = resp.shape[1]
    side = int(round(math.sqrt(n_c)))
    if side * side != n_c:
        return None
    alive = resp.std(0) > 1e-8
    z = (resp - resp.mean(0)) / np.where(alive, resp.std(0), 1.0)
    corr = (z.T @ z) / len(z)
    dist = torus_distance(side)
    off = ~np.eye(n_c, dtype=bool) & alive[:, None] & alive[None]
    edges = np.arange(0.5, dist.max() + 1.0, 1.0)
    centres, curve = [], []
    for lo, hi in zip(np.concatenate([[0.0], edges[:-1]]), edges):
        sel = off & (dist > lo) & (dist <= hi)
        if sel.any():
            centres.append(float(dist[sel].mean()))
            curve.append(float(corr[sel].mean()))
    near, allp = corr[off & (dist <= 1.5)].mean(), corr[off].mean()
    return {"d": centres, "corr": curve, "TI": float(near - allp), "fit": exp_fit(np.array(centres), np.array(curve))}


def exp_fit(d, c):
    """c(d) = a·exp(−d/λ) + b; returns (a, λ, b) or None."""
    try:
        from scipy.optimize import curve_fit
        p, _ = curve_fit(lambda x, a, lam, b: a * np.exp(-x / lam) + b, d, c,
                         p0=(max(c[0] - c[-1], 1e-3), 2.0, c[-1]), bounds=([0, 0.05, -1], [2, 100, 1]), maxfev=5000)
        return [float(v) for v in p]
    except Exception:
        return None


def area_maps(grat, nat, nat_grey, names, grat_index):
    """Per-unit maps on the sheet for one area. grat/nat/nat_grey: [N, C]."""
    n_c = grat.shape[1]
    side = int(round(math.sqrt(n_c)))
    if side * side != n_c:
        return None
    top = topography(grat, grat_index, np.random.default_rng(0), n_shuffle=50)
    g = grat[: grat_index["n_grating"]]
    sfs = np.array([s for o in range(8) for s in range(3) for p in range(4)])
    sf_pref = np.argmax(np.stack([g[sfs == s].mean(0) for s in range(3)]), axis=0)
    animate = np.array([n in ANIMATE for n in names])
    human = np.array([n in HUMAN for n in names])
    diff = nat - nat_grey
    chroma_t = diff.mean(0) / (diff.std(0, ddof=1) / math.sqrt(len(diff)) + 1e-12)
    sh = lambda v: np.asarray(v, dtype=float).reshape(side, side).tolist()
    return {
        "orientation_pref": top["orientation_pref"],
        "orientation_selectivity": top["orientation_selectivity"],
        "sf_pref": sh(sf_pref),
        "chroma_t": sh(chroma_t),
        "animacy_t": sh(welch_t(nat[animate], nat[~animate])),
        "human_t": sh(welch_t(nat[human], nat[~human])),
        "natural_corr_vs_distance": corr_vs_distance(nat),
    }


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_map_grid(results, key, title, path, cmap="RdBu_r", symmetric=True):
    plt = _plt()
    from matplotlib.colors import hsv_to_rgb
    labels = list(results)
    areas = [a for a in results[labels[0]]["maps"] if results[labels[0]]["maps"][a] is not None]
    fig, axes = plt.subplots(len(labels), len(areas), figsize=(3.0 * len(areas), 3.0 * len(labels)), squeeze=False)
    for i, lab in enumerate(labels):
        for j, area in enumerate(areas):
            ax, m = axes[i, j], results[lab]["maps"][area]
            ax.set_xticks([]); ax.set_yticks([])
            if key == "orientation":
                pref, osi = np.array(m["orientation_pref"]), np.array(m["orientation_selectivity"])
                val = np.clip(osi / max(osi.max(), 1e-9), 0.15, 1.0)
                ax.imshow(hsv_to_rgb(np.stack([pref / np.pi, np.ones_like(pref), val], -1)), interpolation="nearest")
            else:
                v = np.array(m[key])
                if symmetric:
                    lim = max(np.nanpercentile(np.abs(v), 98), 1e-6)
                    im = ax.imshow(v, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
                else:
                    im = ax.imshow(v, cmap=cmap, interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_title(f"{lab} {area}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layout(results, path):
    """Per model: V1 | V2 | V4 | IT side by side at the same height (posterior → anterior), for 3 maps."""
    plt = _plt()
    labels = list(results)
    rows = [("animacy_t", "animate (red) vs inanimate (blue)"), ("human_t", "human (red) vs rest"),
            ("chroma_t", "colour (red) vs grey-scale (blue)")]
    fig, axes = plt.subplots(len(labels) * len(rows), 4, figsize=(11, 2.8 * len(labels) * len(rows)), squeeze=False)
    for i, lab in enumerate(labels):
        maps = results[lab]["maps"]
        for r, (key, desc) in enumerate(rows):
            for j, area in enumerate(["V1", "V2", "V4", "IT"]):
                ax = axes[i * len(rows) + r, j]
                ax.set_xticks([]); ax.set_yticks([])
                if maps.get(area) is None:
                    continue
                v = np.array(maps[area][key])
                lim = max(np.nanpercentile(np.abs(v), 98), 1e-6)
                ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.set_title(f"{area}" if j else f"{lab}: {desc}\n{area}", fontsize=8, loc="left" if not j else "center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corr_distance(results, path):
    plt = _plt()
    labels = list(results)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        for k, (area, m) in enumerate(results[lab]["maps"].items()):
            if m is None or m["natural_corr_vs_distance"] is None:
                continue
            c = m["natural_corr_vs_distance"]
            ax.plot(c["d"], c["corr"], "o", ms=3, color=f"C{k}")
            if c["fit"]:
                a, lam, b = c["fit"]
                d = np.linspace(0, max(c["d"]), 100)
                ax.plot(d, a * np.exp(-d / lam) + b, "-", color=f"C{k}", label=f"{area} λ={lam:.1f}")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"{lab}: natural images", fontsize=9)
        ax.set_xlabel("sheet distance"); ax.set_ylabel("tuning correlation"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timecourse(results, path):
    plt = _plt()
    labels = [l for l in results if len(results[l]["timecourse"]["output"]["V1"]) > 1]
    if not labels:
        return False
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        tc = results[lab]["timecourse"]
        for k, area in enumerate(tc["output"]):
            ax.plot(np.arange(1, len(tc["output"][area]) + 1), tc["output"][area], "-", color=f"C{k}", label=f"{area} output")
            ax.plot(np.arange(1, len(tc["drive"][area]) + 1), tc["drive"][area], ":", color=f"C{k}")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"{lab}: topography index over time\n(solid output, dotted drive; natural images)", fontsize=9)
        ax.set_xlabel("timestep"); ax.set_ylabel("TI"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--label", action="append")
    ap.add_argument("--h5", required=True, help="mini-ecoset .h5 (uses its 'val' split and 'categories')")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--signal", choices=["drive", "output"], default="drive", help="which signal the maps use")
    ap.add_argument("--timesteps", type=int, default=None, help="presentation length (default: the model's n_timesteps)")
    ap.add_argument("--out", default="topography_maps")
    args = ap.parse_args()
    labels = args.label or [Path(m).parent.parent.name for m in args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for path, lab in zip(args.model, labels):
        model, cfg = load_trained_model(path, device=device)
        mean, std = cfg["data.normalize"]
        res = int(cfg["data.resolution"])
        T = args.timesteps or int(getattr(model, "n_timesteps", 1))
        grat, grat_index = make_stimuli(res)
        nat, names = load_val_images(args.h5, args.per_class, res)
        r_grat = responses_over_time(model, grat, mean, std, T, device)
        r_nat = responses_over_time(model, nat, mean, std, T, device)
        r_grey = responses_over_time(model, greyscale(nat), mean, std, T, device)
        s = args.signal
        maps = {a: area_maps(r_grat[s][a][-1], r_nat[s][a][-1], r_grey[s][a][-1], names, grat_index) for a in model.layer_names}
        timecourse = {kind: {a: [] for a in model.layer_names} for kind in ("drive", "output")}
        for kind in timecourse:
            for a in model.layer_names:
                for t in range(T):
                    c = corr_vs_distance(r_nat[kind][a][t])
                    timecourse[kind][a].append(c["TI"] if c else float("nan"))
        results[lab] = {"maps": maps, "timecourse": timecourse, "timesteps": T}
        print(lab, "natural-image TI (last step, %s):" % s,
              {a: round(m["natural_corr_vs_distance"]["TI"], 3) for a, m in maps.items() if m})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out / "topography_maps.json").write_text(json.dumps(results))
    plot_map_grid(results, "orientation", "Preferred orientation (hue) and selectivity (brightness)", out / "map_orientation.png")
    plot_map_grid(results, "sf_pref", "Preferred spatial frequency (0 low, 1 mid, 2 high)", out / "map_spatial_frequency.png",
                  cmap="viridis", symmetric=False)
    plot_map_grid(results, "chroma_t", "Chromaticity: colour vs grey-scale (t)", out / "map_chromaticity.png")
    plot_map_grid(results, "animacy_t", "Animate vs inanimate (t)", out / "map_animacy.png")
    plot_map_grid(results, "human_t", "Human (man/woman/child) vs rest (t)", out / "map_human.png")
    plot_layout(results, out / "layout_V1_to_IT.png")
    plot_corr_distance(results, out / "corr_vs_distance_natural.png")
    has_tc = plot_timecourse(results, out / "timecourse_TI.png")
    print("wrote figures to", out, "(incl. timecourse)" if has_tc else "(no recurrent model: no timecourse)")


if __name__ == "__main__":
    main()
