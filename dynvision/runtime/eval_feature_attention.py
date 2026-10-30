"""Feature-based attention and cue priming without training, after Lindsay & Miller (2018, eLife 7:e38105).

Task: "is the cued category in this image?", on composites of mini-ecoset test images.
    merged  two images blended 50/50 (one is the cued category on present trials)
    array   a 2x2 grid of four images (one is the cued category on present trials)
Readout: Lindsay & Miller's choice, one logistic-regression detector per category, trained on the IT
output (spatial mean) of clean val images, category vs the rest. It is applied at every timestep,
which gives the time course for recurrent models. Performance: ROC AUC between present and absent
trials per cued category; effects are paired differences against no attention, per category.

1. Attention gain, g_k = max(0, 1 + β · f_k(cued category)), at every timestep.
   f_k(c) is z-scored across categories and scaled per channel to max |f| = 1, so β is the largest
   relative gain change (real feature-similarity gain is about ±10-30%, Treue & Martinez-Trujillo 1999).
   --values  tuning: channel k's mean response to clean images of c (feature-similarity gain)
             gradient: d logit_c / d (channel k), averaged over clean images of c (Lindsay & Miller's
             gradient values; better below the top layer)
   --sites   output: on the area's output, after the nonlinearity (also reaches the recurrence)
             drive: on the feedforward drive only (the area's conv output, before the time step)
             recurrence: on the lateral recurrent input only (attention gating context; recurrent
             models only)
   --layers  areas that receive attention; several sets separated by ';' (e.g. "V1;V2;V4;IT;V4,IT")
   Variants: gain, and gain+dn: the gain inside a divisive normalization over channels at each
   location (Reynolds & Heeger 2009), y = g·x·(σ + Σx)/(σ + Σg·x), σ = mean pooled activity.
   Negative β (attend away) is the control.

2. Cue priming (--prime), no gain: a clean exemplar of a category is shown for --cue-steps, then
   (after --isi blank steps) the composite for --search-steps. The cue is either the cued category
   (match), another category that is not in the composite (mismatch), or blank. Any match benefit
   comes from the network's own state carrying the cue. A feedforward model with tau = dt has no
   state, so its three conditions must be identical (a built-in control).

Usage:
    python -m dynvision.runtime.eval_feature_attention \\
        --model .../plain_7000/miniecoset/trained.pt --label FF \\
        --model .../R1_7000/miniecoset/trained.pt --label R1 \\
        --h5 .../texture2shape_miniecoset_256px.h5 --sites output,drive,recurrence --prime --out attention_week1
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynvision.utils.model_utils import load_trained_model


# ----------------------------------------------------------------- attention machinery
def _apply_gain(x, gain, renormalize):
    y = x * gain.to(x.dtype)
    if renormalize:
        pool, pool_att = x.sum(1, keepdim=True), y.sum(1, keepdim=True)
        sigma = pool.mean()
        y = y * (sigma + pool) / (sigma + pool_att)
    return y


class AreaOutput(nn.Module):
    """Inserted after each area's output nonlinearity: output gain, gradient capture, per-step recording."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = None          # [B, C, 1, 1] or None
        self.renormalize = False
        self.capture = None       # list of outputs with retained grad (gradient values)
        self.record = None        # list of per-step spatial means (readout time course)

    def forward(self, x):
        if x is None:
            return x
        if self.gain is not None:
            x = _apply_gain(x, self.gain, self.renormalize)
        if self.capture is not None:
            x.retain_grad()
            self.capture.append(x)
        if self.record is not None:
            self.record.append(x.detach().float().mean(dim=(-2, -1)).cpu())
        return x


class GainHook:
    """Forward hook that scales a submodule's output (drive conv or recurrence)."""

    def __init__(self, module) -> None:
        self.gain, self.renormalize = None, False
        self.handle = module.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        if self.gain is None or output is None:
            return output
        return _apply_gain(output, self.gain, self.renormalize)


class Attention:
    def __init__(self, model) -> None:
        ops = model.layer_operations
        if "areaout" not in ops:
            after = "sheetpool" if "sheetpool" in ops else "supralin"
            ops.insert(ops.index(after) + 1, "areaout")
        dev = next(model.parameters()).device
        self.out, self.drive, self.rec = {}, {}, {}
        for name in model.layer_names:
            layer = getattr(model, name)
            mod = AreaOutput().to(dev)
            setattr(model, f"areaout_{name}", mod)
            self.out[name] = mod
            conv = layer.conv2 if getattr(layer, "mid_channels", None) else layer.conv
            self.drive[name] = GainHook(conv)
            if getattr(layer, "recurrence", None) is not None:
                self.rec[name] = GainHook(layer.recurrence)

    def set(self, site=None, gains=None, renormalize=False, sl=slice(None), device=None):
        for d in (self.out, self.drive, self.rec):
            for m in d.values():
                m.gain, m.renormalize = None, False
        if site is None or gains is None:
            return
        target = {"output": self.out, "drive": self.drive, "recurrence": self.rec}[site]
        for name, g in gains.items():
            if name in target:
                target[name].gain = g[sl].to(device)
                target[name].renormalize = renormalize


# ----------------------------------------------------------------- data
class Images:
    def __init__(self, h5_path: str, split: str):
        import h5py
        self.f = h5py.File(h5_path, "r")
        self.data = self.f[split]["data"]
        self.labels = np.asarray(self.f[split]["labels"][:])
        self.n_classes = int(self.labels.max()) + 1

    def get(self, idx, size: int) -> torch.Tensor:
        """uint8 [N, 3, size, size] for the given indices (resized from the stored 256 px)."""
        idx = np.asarray(idx)
        uniq, inverse = np.unique(idx, return_inverse=True)        # h5 needs strictly increasing indices
        out = []
        for i in range(0, len(uniq), 256):
            x = torch.from_numpy(self.data[uniq[i:i + 256].tolist()]).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
            out.append(x.clamp(0, 255).round().to(torch.uint8))
        return torch.cat(out)[torch.from_numpy(inverse)]


def make_trials(img: Images, task: str, categories, n_trials: int, res: int, rng):
    """Composite trials (half present) per cued category, plus cue exemplars for priming:
    a match exemplar (cued category, not one of the composite's images) and a mismatch exemplar
    (a category absent from the composite)."""
    by_class = {c: np.flatnonzero(img.labels == c) for c in range(img.n_classes)}
    n_parts = 2 if task == "merged" else 4
    picks, cues, present, cue_match, cue_mismatch = [], [], [], [], []
    for c in categories:
        others = [k for k in range(img.n_classes) if k != c]
        for t in range(n_trials):
            is_present = t < n_trials // 2
            cls = list(rng.choice(others, size=n_parts, replace=False))
            if is_present:
                cls[rng.integers(n_parts)] = c
            pick = [int(rng.choice(by_class[k])) for k in cls]
            picks.append(pick)
            cues.append(c)
            present.append(is_present)
            cue_match.append(int(rng.choice([i for i in by_class[c] if i not in pick])))
            absent = [k for k in others if k not in cls]
            cue_mismatch.append(int(rng.choice(by_class[int(rng.choice(absent))])))
    psize = res if task == "merged" else res // 2
    part = img.get(np.array(picks).reshape(-1), psize).reshape(len(picks), n_parts, 3, psize, psize)
    if task == "merged":
        comp = part.float().mean(1).round().to(torch.uint8)
    else:
        comp = torch.cat([torch.cat([part[:, 0], part[:, 1]], -1), torch.cat([part[:, 2], part[:, 3]], -1)], -2)
    return comp, np.array(cues), np.array(present), img.get(cue_match, res), img.get(cue_mismatch, res)


# ----------------------------------------------------------------- model runs
def run(model, att, segments, mean, std, device, batch, site=None, gains=None, renorm=False):
    """Present a sequence of segments [(uint8 images [N,3,H,W] or None = blank, n_steps), ...].
    Returns the IT output's spatial mean at every step: [N, T_total, C]."""
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    ref_imgs = next(s[0] for s in segments if s[0] is not None)
    it = att.out[model.layer_names[-1]]
    states = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(0, len(ref_imgs), batch):
            frames = []
            for imgs, steps in segments:
                if imgs is None:                                              # blank = dataset mean after normalization
                    x = torch.zeros(ref_imgs[i:i + batch].shape, device=device)
                else:
                    x = (imgs[i:i + batch].to(device).float() / 255.0 - mean_t) / std_t
                frames.append(x[:, None].expand(-1, steps, -1, -1, -1))
            att.set(site, gains, renorm, slice(i, i + batch), device)
            it.record = []
            model(torch.cat(frames, dim=1).contiguous())
            states.append(torch.stack(it.record, dim=1))
            it.record = None
    att.set()
    return torch.cat(states)


def gradient_values(model, att, img, idx, mean, std, T, device, batch) -> dict:
    """d logit_c / d (area output channel k), summed over space and steps, averaged over images of c."""
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    labels = img.labels[idx]
    acc = {n: torch.zeros(img.n_classes, getattr(model, n).out_channels) for n in model.layer_names}
    cnt = torch.zeros(img.n_classes)
    for i in range(0, len(idx), batch):
        x = (img.get(idx[i:i + batch], int(model.dim_y)).to(device).float() / 255.0 - mean_t) / std_t
        y = torch.from_numpy(labels[i:i + batch]).to(device)
        for m in att.out.values():
            m.capture = []
        model.zero_grad(set_to_none=True)
        out = model(x[:, None].repeat(1, T, 1, 1, 1))
        out = out if out.dim() == 3 else out[:, None]
        out[torch.arange(len(y)), -1, y].sum().backward()
        for name, m in att.out.items():
            g = sum(h.grad.float().sum(dim=(-2, -1)) for h in m.capture if h.grad is not None)
            acc[name].index_add_(0, y.cpu(), g.detach().cpu())
            m.capture = None
        cnt.index_add_(0, y.cpu(), torch.ones(len(y)))
    model.zero_grad(set_to_none=True)
    return {n: a / cnt.clamp_min(1)[:, None] for n, a in acc.items()}


def auc_per_category(score, present, cues) -> np.ndarray:
    out = []
    for c in np.unique(cues):
        s, p = score[cues == c], present[cues == c]
        pos, neg = s[p], s[~p]
        if len(pos) and len(neg):
            ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
            out.append((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return np.array(out)


def summarize(a, a_ref):
    d = a - a_ref
    sem = lambda v: float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"auc": float(a.mean()), "auc_sem": sem(a), "delta_auc": float(d.mean()), "delta_sem": sem(d)}


# ----------------------------------------------------------------- one model
def evaluate(path, label, args, img_tune, img_test, device):
    from sklearn.linear_model import LogisticRegression

    model, cfg = load_trained_model(path, device=device)
    model.eval()
    att = Attention(model)
    mean, std = cfg["data.normalize"]
    res = int(cfg["data.resolution"])
    T = args.timesteps or int(getattr(model, "n_timesteps", 1))
    rng = np.random.default_rng(args.seed)                      # same trials for every model
    result = {"label": label, "path": str(path), "timesteps": T, "values": args.values, "tasks": {}}
    print(f"[{label}] T={T}, recurrence in: {sorted(att.rec) or 'none'}", flush=True)

    # clean val images: IT detectors and attention values per area
    tune_idx = np.concatenate([rng.choice(np.flatnonzero(img_tune.labels == c), args.n_tune, replace=False)
                               for c in range(img_tune.n_classes)])
    labels_tune = img_tune.labels[tune_idx]
    x_tune = img_tune.get(tune_idx, res)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    rec = {n: [] for n in model.layer_names}
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(0, len(x_tune), args.batch):
            for name in model.layer_names:
                att.out[name].record = []
            x = (x_tune[i:i + args.batch].to(device).float() / 255.0 - mean_t) / std_t
            model(x[:, None].repeat(1, T, 1, 1, 1))
            for name in model.layer_names:
                rec[name].append(att.out[name].record[-1])                    # last step
    states = {}
    for name in model.layer_names:
        att.out[name].record = None
        states[name] = torch.cat(rec[name])
    if args.values == "gradient":
        raw = gradient_values(model, att, img_tune, tune_idx, mean, std, T, device, max(1, args.batch // 2))
    else:
        raw = {n: torch.stack([s[labels_tune == c].mean(0) for c in range(img_tune.n_classes)]) for n, s in states.items()}
    values = {}
    for name, m in raw.items():
        z = (m - m.mean(0)) / (m.std(0) + 1e-6)                               # z-score across categories
        values[name] = z / z.abs().amax(0).clamp_min(1e-6)                     # per channel max |f| = 1

    cats = rng.choice(img_test.n_classes, size=min(args.categories, img_test.n_classes), replace=False)
    feats = states[model.layer_names[-1]].numpy()
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    det = {}
    for c in cats:
        clf = LogisticRegression(class_weight="balanced", C=args.readout_c, max_iter=2000)
        clf.fit((feats - mu) / sd, labels_tune == c)
        det[int(c)] = (clf.coef_[0], clf.intercept_[0])

    def scores(it_states, cues):                                              # [N, T] detector score per step
        f = (it_states.numpy() - mu) / sd
        out = np.zeros(f.shape[:2])
        for c in np.unique(cues):
            sel = cues == c
            w, b = det[int(c)]
            out[sel] = f[sel] @ w + b
        return out

    for task in args.tasks.split(","):
        comp, cues, present, cue_m, cue_x = make_trials(img_test, task, cats, args.n_trials, res, rng)
        tr = {"attention": {}, "priming": {}}

        def record(key, it_states, ref):
            sc = scores(it_states, cues)
            auc_t = [float(auc_per_category(sc[:, t], present, cues).mean()) for t in range(sc.shape[1])]
            a = auc_per_category(sc[:, -1], present, cues)
            entry = summarize(a, a if ref is None else ref)
            entry["auc_by_timestep"] = auc_t
            print(f"[{label}] {task:>6} {key:>34}: AUC {entry['auc']:.3f}  ΔAUC {entry['delta_auc']:+.4f} ± {entry['delta_sem']:.4f}", flush=True)
            return a, entry

        a_none, tr["attention"]["none"] = record("none", run(model, att, [(comp, T)], mean, std, device, args.batch), None)
        for site in args.sites.split(","):
            if site == "recurrence" and not att.rec:
                continue
            for layerset in args.layers.split(";"):
                layers = [l for l in layerset.split(",") if l in model.layer_names]
                if site == "recurrence":
                    layers = [l for l in layers if l in att.rec]
                if not layers:
                    continue
                for variant in (("gain",) if site == "recurrence" else ("gain", "gain+dn")):
                    for beta in [b for b in args.betas if b != 0]:
                        gains = {n: (1 + beta * values[n][cues]).clamp_min(0)[:, :, None, None] for n in layers}
                        key = f"{site}|{','.join(layers)}|{variant}|{beta:g}"
                        st = run(model, att, [(comp, T)], mean, std, device, args.batch, site, gains, variant == "gain+dn")
                        _, tr["attention"][key] = record(key, st, a_none)

        if args.prime:
            def seq(cue):
                s = [(cue, args.cue_steps)]
                if args.isi:
                    s.append((None, args.isi))
                return s + [(comp, args.search_steps)]
            a_blank, tr["priming"]["blank"] = record("prime blank", run(model, att, seq(None), mean, std, device, args.batch), None)
            for name, cue in (("match", cue_m), ("mismatch", cue_x)):
                _, tr["priming"][name] = record(f"prime {name}", run(model, att, seq(cue), mean, std, device, args.batch), a_blank)
            tr["priming"]["layout"] = {"cue_steps": args.cue_steps, "isi": args.isi, "search_steps": args.search_steps}
        result["tasks"][task] = tr
    return result


# ----------------------------------------------------------------- plots
def plot(results, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = args.tasks.split(",")
    sites = args.sites.split(",")
    layersets = args.layers.split(";")
    betas = sorted(b for b in args.betas if b != 0)
    rows = len(sites) * len(layersets) + 1 + (1 if args.prime else 0)
    fig, axes = plt.subplots(rows, len(tasks), figsize=(5.4 * len(tasks), 3.4 * rows), squeeze=False)
    r_i = 0
    for site in sites:
        for layerset in layersets:
            for j, task in enumerate(tasks):
                ax = axes[r_i, j]
                for i, r in enumerate(results):
                    att = r["tasks"][task]["attention"]
                    for variant, ls in (("gain", "-"), ("gain+dn", "--")):
                        keys = [f"{site}|{layerset}|{variant}|{b:g}" for b in betas]
                        if not all(k in att for k in keys):
                            continue
                        xs, ys, es = zip(*sorted(zip([0] + betas, [0.0] + [att[k]["delta_auc"] for k in keys],
                                                     [0.0] + [att[k]["delta_sem"] for k in keys])))
                        ax.errorbar(xs, ys, yerr=es, ls=ls, marker="o", ms=3, capsize=2, color=f"C{i}", label=f"{r['label']} {variant}")
                ax.axhline(0, color="0.7", lw=0.8)
                ax.set_title(f"{task}: {site} gain at {layerset}", fontsize=9)
                ax.set_xlabel("β (negative = attend away)", fontsize=8); ax.set_ylabel("Δ AUC (paired)", fontsize=8)
                ax.legend(fontsize=6)
            r_i += 1
    bpos = max(betas)
    for j, task in enumerate(tasks):                                               # time courses
        ax = axes[r_i, j]
        for i, r in enumerate(results):
            if r["timesteps"] <= 1:
                continue
            att = r["tasks"][task]["attention"]
            ax.plot(np.arange(1, r["timesteps"] + 1), att["none"]["auc_by_timestep"], ":", color=f"C{i}", label=f"{r['label']} none")
            for site, ls in (("output", "-"), ("drive", "--"), ("recurrence", "-.")):
                k = f"{site}|{layersets[0]}|gain|{bpos:g}"
                if k in att:
                    ax.plot(np.arange(1, r["timesteps"] + 1), att[k]["auc_by_timestep"], ls, color=f"C{i}", label=f"{r['label']} {site} β={bpos:g}")
        ax.set_title(f"{task}: time course, attention at {layersets[0]}", fontsize=9)
        ax.set_xlabel("timestep", fontsize=8); ax.set_ylabel("AUC", fontsize=8); ax.legend(fontsize=6)
    r_i += 1
    if args.prime:
        for j, task in enumerate(tasks):
            ax = axes[r_i, j]
            for i, r in enumerate(results):
                pr = r["tasks"][task]["priming"]
                for name, ls in (("blank", ":"), ("match", "-"), ("mismatch", "--")):
                    ax.plot(np.arange(1, len(pr[name]["auc_by_timestep"]) + 1), pr[name]["auc_by_timestep"], ls, color=f"C{i}",
                            label=f"{r['label']} {name} (Δ {pr[name]['delta_auc']:+.3f})")
            lay = results[0]["tasks"][task]["priming"]["layout"]
            ax.axvspan(0.5, lay["cue_steps"] + 0.5, color="0.92")
            ax.set_title(f"{task}: cue priming (grey = cue frames)", fontsize=9)
            ax.set_xlabel("timestep", fontsize=8); ax.set_ylabel("AUC", fontsize=8); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--label", action="append")
    ap.add_argument("--h5", required=True, help="mini-ecoset h5 (texture2shape_miniecoset_256px.h5)")
    ap.add_argument("--tune-split", default="val")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--n-tune", type=int, default=20, help="clean val images per category (detectors, values)")
    ap.add_argument("--categories", type=int, default=40)
    ap.add_argument("--n-trials", type=int, default=20, help="trials per cued category and task (half present)")
    ap.add_argument("--tasks", default="merged,array")
    ap.add_argument("--values", choices=("tuning", "gradient"), default="gradient")
    ap.add_argument("--sites", default="output", help="comma list of output,drive,recurrence")
    ap.add_argument("--layers", default="V4,IT", help="area sets separated by ';', e.g. 'V1;V2;V4;IT;V4,IT'")
    ap.add_argument("--betas", default="0.1,0.25,0.5,1,-0.5", help="negative = attend away (control)")
    ap.add_argument("--prime", action="store_true", help="also run cue priming (match / mismatch / blank cue)")
    ap.add_argument("--cue-steps", type=int, default=8)
    ap.add_argument("--isi", type=int, default=0)
    ap.add_argument("--search-steps", type=int, default=16)
    ap.add_argument("--readout-c", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=None, help="steps per presentation (default: the model's)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="feature_attention")
    args = ap.parse_args()
    args.betas = [float(b) for b in args.betas.split(",")]
    labels = args.label or [Path(m).parent.parent.name for m in args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tune, img_test = Images(args.h5, args.tune_split), Images(args.h5, args.test_split)

    results = [evaluate(p, lab, args, img_tune, img_test, device) for p, lab in zip(args.model, labels)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    plot(results, args, out.with_suffix(".png"))
    print("wrote", out.with_suffix(".json"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
