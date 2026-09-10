"""
DynVision to Model-vs-Human Cue-Conflict Evaluation Bridge.

Evaluates a trained DynVision recurrent model (e.g. DyRCNNx4) on the Bethge Lab
Model-vs-Human Cue-Conflict Benchmark across all recurrent timesteps t = 0 ... T-1,
recording predictions and generating the shape-bias comparison plot against human observers.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Ensure ModelVsHuman and DynVision are on path
MODELVSHUMAN_DIR = Path(os.environ.get("MODELVSHUMANDIR", "/home/glenn/dynvision/model-vs-human-master"))
os.environ["MODELVSHUMANDIR"] = str(MODELVSHUMAN_DIR)
sys.path.insert(0, str(MODELVSHUMAN_DIR))

from dynvision.data.miniecoset_dataset import (
    GEIRHOS_16_CLASSES,
    ECOSET_TO_GEIRHOS_ALIASES,
    GEIRHOS_TO_ECOSET_ALIASES,
    get_geirhos_mapping,
    MINIECOSET_MEAN,
    MINIECOSET_STD,
    get_default_transforms,
)


class CueConflictDataset(torch.utils.data.Dataset):
    """Loads the 1,280 Geirhos cue-conflict images from model-vs-human datasets/cue-conflict."""
    def __init__(self, dataset_dir: Path, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.samples = []
        
        for p in sorted(self.dataset_dir.glob("*/*.png")):
            # Filename convention: <shape>_<texture>_<id>.png or similar
            fname = p.stem
            parts = fname.split("_")
            shape_cat = parts[0]
            texture_cat = parts[1] if len(parts) > 1 else shape_cat
            self.samples.append({
                "path": str(p),
                "filename": p.name,
                "shape_cat": shape_cat,
                "texture_cat": texture_cat,
                "dir_cat": p.parent.name
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, item["shape_cat"], item["texture_cat"], item["filename"]


def run_temporal_cue_conflict_evaluation(
    model: torch.nn.Module,
    categories_112: list,
    output_dir: Path,
    cue_conflict_dir: Path = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    model_name_prefix: str = "DyRCNNx4",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cue_conflict_dir = cue_conflict_dir or (MODELVSHUMAN_DIR / "datasets" / "cue-conflict")
    if not cue_conflict_dir.exists():
        raise FileNotFoundError(f"Cue-conflict dataset not found at {cue_conflict_dir}")

    transform = get_default_transforms(img_size=224, split="val")
    ds = CueConflictDataset(cue_conflict_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 1. Build projection from 112 classes to 16 Geirhos classes
    mapping_info = get_geirhos_mapping(categories_112)
    g16_indices = torch.tensor(mapping_info["geirhos_to_112_indices"], device=device)

    model.eval()
    model.to(device)

    # We will accumulate records per timestep
    # shape of records_by_t[t] = list of dicts
    all_predictions = {}

    print(f"\nRunning cue-conflict evaluation across recurrent timesteps...")
    with torch.no_grad():
        for batch_imgs, shapes, textures, filenames in tqdm(loader, desc="Batches"):
            batch_imgs = batch_imgs.to(device)
            # Model forward rollout
            # DynVision DyRCNN forward returns logits for timesteps: (B, T, 112) or (B, 112)
            outputs = model(batch_imgs)
            
            if isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
                # (Batch, Timesteps, Classes)
                B, T, C = outputs.shape
                temporal_logits = outputs
            elif isinstance(outputs, (list, tuple)):
                # List of (B, C) per timestep
                T = len(outputs)
                temporal_logits = torch.stack(outputs, dim=1)
            else:
                # Single timestep output (B, C)
                T = 1
                temporal_logits = outputs.unsqueeze(1)

            for t in range(T):
                if t not in all_predictions:
                    all_predictions[t] = []

                # Extract logits at timestep t restricted to 16 Geirhos classes
                logits_112 = temporal_logits[:, t, :]  # (B, 112)
                logits_16 = logits_112[:, g16_indices]  # (B, 16)
                pred_indices = torch.argmax(logits_16, dim=1).cpu().numpy()

                for i in range(len(filenames)):
                    pred_class = GEIRHOS_16_CLASSES[pred_indices[i]]
                    shape_gt = shapes[i]
                    texture_gt = textures[i]
                    is_shape = (pred_class == shape_gt)
                    is_texture = (pred_class == texture_gt)

                    all_predictions[t].append({
                        "filename": filenames[i],
                        "shape": shape_gt,
                        "texture": texture_gt,
                        "prediction": pred_class,
                        "correct_shape": is_shape,
                        "correct_texture": is_texture
                    })

    # Compute and report summary per timestep
    summary_rows = []
    print("\n" + "=" * 65)
    print(f"{'Timestep':^10} | {'Shape Corr':^12} | {'Texture Corr':^12} | {'Shape Bias':^15}")
    print("=" * 65)

    for t in sorted(all_predictions.keys()):
        df_t = pd.DataFrame(all_predictions[t])
        n_shape = int(df_t["correct_shape"].sum())
        n_texture = int(df_t["correct_texture"].sum())
        total_decisions = n_shape + n_texture
        shape_bias = (n_shape / total_decisions) if total_decisions > 0 else float("nan")

        summary_rows.append({
            "timestep": t,
            "correct_shape": n_shape,
            "correct_texture": n_texture,
            "total_decisions": total_decisions,
            "shape_bias": shape_bias,
        })
        print(f"{t:^10} | {n_shape:^12} | {n_texture:^12} | {shape_bias:^15.4f}")

        # Save individual timestep CSV in model-vs-human raw-data format
        csv_path = output_dir / f"{model_name_prefix}_t{t}_cue-conflict.csv"
        df_t.to_csv(csv_path, index=False)

    print("=" * 65)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"{model_name_prefix}_shape_bias_temporal_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt or .ckpt")
    parser.add_argument("--categories_h5", type=str, default="/home/glenn/DynVision_Working/data/raw/mini-ecoset/texture2shape_miniecoset_256px.h5")
    parser.add_argument("--output_dir", type=str, default="/home/glenn/DynVision_Working/reports/model_vs_human")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    import h5py
    with h5py.File(args.categories_h5, "r") as f:
        categories_112 = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in f["categories"][:]]

    model = torch.load(args.model_path)
    if isinstance(model, dict) and "state_dict" in model:
        # Load lightning state_dict if passed .ckpt
        from dynvision.models.dyrcnn import DyRCNNx4
        net = DyRCNNx4(n_classes=112, recurrence_type="self")
        net.load_state_dict(model["state_dict"])
        model = net

    run_temporal_cue_conflict_evaluation(
        model=model,
        categories_112=categories_112,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size
    )
