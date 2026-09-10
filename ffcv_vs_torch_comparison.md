# PyTorch vs FFCV Performance Differences

This document outlines every mechanical and architectural difference between the PyTorch dataloader and the FFCV dataloader within the DynVision codebase. If a model trained with PyTorch outperforms the same model trained with FFCV, it is due to one or a combination of the following factors.

## 1. Data Augmentations (The Primary Factor)
The configurations for Torch and FFCV apply entirely different geometric augmentations, which is the most likely cause of performance discrepancies.

*   **PyTorch Pipeline** (from `torch.train.base`):
    *   `RandomRotation(10)`
    *   `RandomAffine(0, translate=(0.1, 0.1))` (Translates by up to 10%, which is ~3.2 pixels for CIFAR-10)
    *   `RandomHorizontalFlip()`
    *   `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
*   **FFCV Pipeline** (from `ffcv.train.cifar10`):
    *   `RandomHorizontalFlip()`
    *   `RandomTranslate(padding=4, fill=(0, 0, 0))` (Translates by up to 4 pixels)
    *   `RandomBrightness(0.2)`
    *   `RandomContrast(0.2)`
    *   `RandomSaturation(0.2)`

**Impact**: PyTorch applies `RandomRotation(10)` and `RandomAffine`, while FFCV only applies `RandomTranslate(padding=4)`. Rotation is a highly effective regularizer that prevents overfitting and improves generalization on datasets like CIFAR-10. FFCV does not natively support rotation, hence it was omitted.

## 2. Order of Operations & Data Range `[0, 1]` vs `[0, 255]`
While the mathematical normalization (subtracting mean and dividing by standard deviation) is identical between both pipelines, the order in which augmentations are applied relative to the data type differs:

*   **PyTorch**: Images are converted to `[0, 1]` floats via `ToTensor()`. Augmentations (Color Jitter, Rotation) are calculated and applied on floating-point values, preserving high precision before final normalization.
*   **FFCV**: Images remain as `[0, 255]` 8-bit integers (`uint8`) throughout the augmentation pipeline to maximize speed. Color jitter and translation are applied in 8-bit space (which introduces minor clipping and rounding errors). Normalization is applied at the very end using scaled constants (`mean * 255`, `std * 255`) just before conversion to `float16/float32`.

**Impact**: Applying contrast and saturation in 8-bit integer space (FFCV) vs 32-bit float space (Torch) results in subtle histogram differences. 

## 3. The Color Jitter Implementations
*   **PyTorch**: `ColorJitter` relies on `PIL` or Torchvision's native C++ backend which smoothly interpolates hue, saturation, and brightness.
*   **FFCV**: Uses a custom, heavily optimized C++ backend for `RandomBrightness`, `RandomContrast`, and `RandomSaturation` designed for maximum throughput.

**Impact**: FFCV's color jitter operations are approximations optimized for speed. If a model is highly sensitive to color distributions (like small feedforward CNNs on CIFAR-10), PyTorch's higher-fidelity jitter might yield slightly better representations.

## 4. Batch Sampling Randomness
*   **PyTorch**: `RandomSampler` draws indices uniformly at random across the entire dataset.
*   **FFCV**: Uses a chunked reading mechanism. To maximize disk I/O, FFCV groups the dataset into pages and reads blocks of data at once, shuffling *within* those constraints.

**Impact**: PyTorch batches have mathematically perfect uniform randomness. FFCV batches have a slightly different variance profile due to block-reading. This can affect the gradient noise in the first few epochs and how Batch Normalization tracks running statistics.

## Conclusion
If you want FFCV to perfectly match PyTorch's performance, the primary gap to bridge is the **data augmentations**. Because FFCV doesn't natively support `RandomRotation`, we would need to implement the fallback mechanism I previously proposed (wrapping PyTorch's `RandomRotation` in FFCV's `ModuleWrapper`) to align the two pipelines exactly.
