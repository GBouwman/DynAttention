#!/usr/bin/env python3
"""
Ultra-Fast Selective Range-Request Downloader for 112-Class Mini-Ecoset (256x256)
- C-accelerated PKZIP decryption (releases GIL, microsecond speed)
- Thread-local Keep-Alive connection pooling with proactive 80-request recycling
  (respects Apache 'Keep-Alive: max=100' server limit without connection resets)
- Bounded 1,000-image batching (caps RAM at ~200 MB and OS handles at < 1,000)
- Auto-resume engine (skips already-downloaded images and splits upon interruption)
- Immediate batch flushing to disk (crash-proof)
"""

import io
import os
import sys

# Disable HDF5 network file locking for Windows UNC / WSL paths
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gc
import time
import json
import zlib
import struct
import ctypes
import zipfile
import threading
import subprocess
import argparse
import urllib.request
import requests
from requests.adapters import HTTPAdapter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

URL = "https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip"
PASSWORD = b"ecoset_msjkk"

# Default 100 base classes from miniecoset_h5_gen.py
MINIECOSET_100 = [
    'man', 'woman', 'child', 'monkey', 'elephant', 'dog', 'pig', 'cat', 'tiger', 'leopard',
    'zebra', 'wolf', 'camel', 'kangaroo', 'deer', 'rhino', 'cow', 'goat', 'mouse', 'squirrel',
    'bird', 'snake', 'turtle', 'frog', 'spider', 'rabbit', 'alligator', 'cockroach', 'chicken',
    'wasp', 'butterfly', 'snail', 'jellyfish', 'worm', 'grasshopper', 'lizard', 'mosquito',
    'lobster', 'ant', 'beetle', 'seashell', 'fire', 'apple', 'flower', 'chestnut', 'egg',
    'lemon', 'banana', 'fern', 'cactus', 'corn', 'lettuce', 'grape', 'rice', 'onion', 'oyster',
    'aloe', 'parsley', 'bubble', 'cherry', 'phone', 'gun', 'book', 'ball', 'clock', 'bag',
    'camera', 'cup', 'key', 'pizza', 'bell', 'toilet', 'knife', 'hamburger', 'axe', 'mailbox',
    'lamp', 'bread', 'shovel', 'spoon', 'bed', 'table', 'ship', 'bus', 'bridge', 'computer',
    'train', 'truck', 'refrigerator', 'airplane', 'lighthouse', 'piano', 'television', 'bicycle',
    'windmill', 'barrel', 'canoe', 'cannon', 'hut', 'helicopter'
]

# 12 additional classes to cover cue-conflict and category balance
ADDITIONAL_12 = [
    'bear', 'boat', 'bottle', 'car', 'chair', 'stove', 'typewriter',
    'horse', 'house', 'tree', 'lion', 'couch'
]

TARGET_112 = MINIECOSET_100 + ADDITIONAL_12

# ----------------- C-Accelerated PKZIP Decryptor -----------------
C_CODE = """
#include <stdint.h>
#include <string.h>

static uint32_t crctable[256];
static int crctable_init = 0;

static void init_crctable(void) {
    for (int i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            if (c & 1) {
                c = 0xedb88320L ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crctable[i] = c;
    }
    crctable_init = 1;
}

static inline uint32_t crc32_step(uint32_t key, unsigned char c) {
    return crctable[(key ^ c) & 0xff] ^ (key >> 8);
}

static inline void update_keys(uint32_t *key0, uint32_t *key1, uint32_t *key2, unsigned char c) {
    *key0 = crc32_step(*key0, c);
    *key1 = (*key1 + (*key0 & 0xff)) * 134775813L + 1;
    *key2 = crc32_step(*key2, (unsigned char)(*key1 >> 24));
}

void init_keys(const char *password, int pass_len, uint32_t *key0, uint32_t *key1, uint32_t *key2) {
    if (!crctable_init) {
        init_crctable();
    }
    *key0 = 305419896L;
    *key1 = 591751049L;
    *key2 = 878082192L;
    for (int i = 0; i < pass_len; i++) {
        update_keys(key0, key1, key2, (unsigned char)password[i]);
    }
}

void decrypt_data(unsigned char *data, int len, const char *password, int pass_len) {
    uint32_t key0, key1, key2;
    init_keys(password, pass_len, &key0, &key1, &key2);
    for (int i = 0; i < 12 && i < len; i++) {
        unsigned short k = key2 | 2;
        unsigned char c = data[i] ^ (unsigned char)(((uint32_t)k * (k ^ 1)) >> 8);
        update_keys(&key0, &key1, &key2, c);
    }
    for (int i = 12; i < len; i++) {
        unsigned short k = key2 | 2;
        unsigned char c = data[i] ^ (unsigned char)(((uint32_t)k * (k ^ 1)) >> 8);
        data[i - 12] = c;
        update_keys(&key0, &key1, &key2, c);
    }
}
"""

def load_c_lib():
    if sys.platform == "win32":
        so_path = Path(__file__).resolve().parent / "libzipcrypto.dll"
        if so_path.exists():
            lib = ctypes.CDLL(str(so_path))
            lib.decrypt_data.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            lib.decrypt_data.restype = None
            return lib
        return None
    else:
        so_path = Path("/home/glenn/dynvision/libzipcrypto.so")
        if not so_path.exists():
            c_src = Path("/tmp/zipcrypto.c")
            c_src.write_text(C_CODE)
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", str(c_src), "-o", str(so_path)], check=True)
        lib = ctypes.CDLL(str(so_path))
        lib.decrypt_data.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.decrypt_data.restype = None
        return lib

CLIB = load_c_lib()

class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP Range file object for reading ZIP Central Directory."""
    def __init__(self, url: str, buffer_size: int = 1024 * 1024 * 8):
        self.url = url
        self.buffer_size = buffer_size
        self.pos = 0
        req = urllib.request.Request(self.url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get("Content-Length", 0))
            self.accept_ranges = resp.headers.get("Accept-Ranges", "") == "bytes"
        self.cache_start = 0
        self.cache = b""

    def seekable(self): return True
    def readable(self): return True
    def writable(self): return False
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.length + offset
        self.pos = max(0, min(self.pos, self.length))
        return self.pos

    def _fetch_range(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url)
        req.add_header("Range", f"bytes={start}-{end}")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    def readinto(self, b):
        size = len(b)
        if size == 0 or self.pos >= self.length: return 0
        if self.cache_start <= self.pos < self.cache_start + len(self.cache):
            offset = self.pos - self.cache_start
            chunk = self.cache[offset:offset + size]
            b[:len(chunk)] = chunk
            self.pos += len(chunk)
            return len(chunk)
        fetch_end = min(self.length - 1, self.pos + max(self.buffer_size, size) - 1)
        self.cache_start = self.pos
        self.cache = self._fetch_range(self.pos, fetch_end)
        chunk = self.cache[:size]
        b[:len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)

    def read(self, size=-1):
        if size == -1 or size is None: size = self.length - self.pos
        b = bytearray(size)
        n = self.readinto(b)
        return bytes(b[:n])


def decode_zip_entry_c(raw_header_bytes: bytes, password: bytes) -> bytes:
    """Decrypt and decompress a raw zip entry directly from bytes in C."""
    magic = raw_header_bytes[:4]
    if magic != b"PK\x03\x04":
        raise ValueError(f"Invalid local header signature: {magic}")
    version, flags, method, time, date, crc, comp_size, uncomp_size, fn_len, extra_len = struct.unpack(
        "<HHHHHIIIHH", raw_header_bytes[4:30]
    )
    data_start = 30 + fn_len + extra_len
    raw_payload = raw_header_bytes[data_start : data_start + comp_size]
    
    if flags & 0x1:
        if CLIB is not None:
            buf = bytearray(raw_payload)
            c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
            CLIB.decrypt_data(c_buf, len(buf), password, len(password))
            decrypted_payload = buf[:len(buf) - 12]
        else:
            decrypter = zipfile._ZipDecrypter(password)
            decrypted_payload = decrypter(raw_payload)[12:]
    else:
        decrypted_payload = raw_payload
        
    if method == 8: return zlib.decompress(decrypted_payload, -15)
    elif method == 0: return bytes(decrypted_payload)
    else: raise ValueError(f"Unsupported compression method: {method}")


# ----------------- Thread-Local Socket Management -----------------
_thread_local = threading.local()

def get_thread_session() -> requests.Session:
    """
    Returns a thread-local requests Session with connection recycling.
    Apache servers have 'Keep-Alive: max=100'. We proactively recycle every 80 requests
    to avoid stale socket resets, closed connection errors, and TLS renegotiation freezes.
    """
    if not hasattr(_thread_local, "session") or _thread_local.session is None:
        _thread_local.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=3)
        _thread_local.session.mount("https://", adapter)
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_local.req_count = 0
        
    _thread_local.req_count += 1
    if _thread_local.req_count >= 80:
        _thread_local.session.close()
        _thread_local.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=3)
        _thread_local.session.mount("https://", adapter)
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_local.req_count = 0
        
    return _thread_local.session


def fetch_and_decode_image(url: str, zinfo: zipfile.ZipInfo, img_size: int, max_retries: int = 3) -> np.ndarray:
    """Fetch an individual image using thread-local connection, C-decrypt and resize."""
    start = zinfo.header_offset
    end = start + 30 + len(zinfo.filename) + 128 + 12 + zinfo.compress_size
    headers = {"Range": f"bytes={start}-{end}"}
    session = get_thread_session()
    
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=headers, timeout=(5, 12))
            resp.raise_for_status()
            decompressed = decode_zip_entry_c(resp.content, PASSWORD)
            with Image.open(io.BytesIO(decompressed)) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size != (img_size, img_size):
                    img_rgb = img_rgb.resize((img_size, img_size), Image.Resampling.BILINEAR)
                return np.array(img_rgb, dtype=np.uint8)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed {zinfo.filename} after {max_retries} attempts: {e}")
            time.sleep(0.2 * (attempt + 1))


def get_default_working_dir() -> Path:
    """Resolve default working raw data directory."""
    if sys.platform == "win32":
        wsl_unc = Path(r"\\wsl.localhost\Ubuntu\home\glenn\DynVision_Working\data\raw\mini-ecoset")
        if wsl_unc.exists():
            return wsl_unc
        wsl_alt = Path(r"\\wsl$\Ubuntu\home\glenn\DynVision_Working\data\raw\mini-ecoset")
        if wsl_alt.exists():
            return wsl_alt
        return Path.cwd()
    wsl_working = Path("/home/glenn/DynVision_Working/data/raw/mini-ecoset")
    if wsl_working.parent.parent.exists():
        return wsl_working
    return Path.home() / "DynVision_Working" / "data" / "raw" / "mini-ecoset"


def parse_args():
    parser = argparse.ArgumentParser(description="Selective 112-Class Mini-Ecoset 256px Downloader (C-Accelerated, Resumable)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_working_dir(),
        help="Directory to save the resulting .h5 dataset (default: auto)",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Target image size (default: 256)")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel worker threads (default: 32)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for writes and flush (default: 1000)")
    parser.add_argument("--n-train", type=int, default=2350, help="Train images per class (default: 2350)")
    parser.add_argument("--n-val", type=int, default=50, help="Val images per class (default: 50)")
    parser.add_argument("--n-test", type=int, default=50, help="Test images per class (default: 50)")
    parser.add_argument("--dataset-name", type=str, default="texture2shape_miniecoset", help="Output dataset basename")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = output_dir / f"{args.dataset_name}_{args.image_size}px.h5"

    print(f"=== Selective 112-Class Mini-Ecoset Downloader (C-Accelerated {args.image_size}x{args.image_size}) ===", flush=True)
    print(f"Output Target: {output_h5}", flush=True)
    print(f"Workers: {args.workers} (Thread-local Keep-Alive pools)", flush=True)
    print(f"Batch Size: {args.batch_size} images per flush", flush=True)
    print(f"C Decryptor Loaded: {CLIB is not None}", flush=True)
    print(f"Images per class -> Train: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}", flush=True)

    # 1. Connect and parse Central Directory
    print(f"\nConnecting to remote archive {URL} ...", flush=True)
    f = HTTPRangeFile(URL)
    zf = zipfile.ZipFile(f)
    print(f"Loaded ZIP Central Directory ({len(zf.filelist)} total entries).", flush=True)

    # 2. Build folder-to-files mapping for target 112 classes
    split_class_entries = {'train': {}, 'val': {}, 'test': {}}
    acceptable_exts = ('.jpg', '.jpeg', '.png')

    for zinfo in zf.filelist:
        fname = zinfo.filename
        if zinfo.is_dir() or not fname.lower().endswith(acceptable_exts):
            continue
        parts = fname.split('/')
        if len(parts) >= 3:
            split, folder = parts[0], parts[1]
            if split in split_class_entries:
                cat_name = folder.split('_', 1)[1] if '_' in folder else folder
                if cat_name in TARGET_112:
                    if cat_name not in split_class_entries[split]:
                        split_class_entries[split][cat_name] = []
                    split_class_entries[split][cat_name].append(zinfo)

    print("\nTarget class counts in zip index:", flush=True)
    for split in ('train', 'val', 'test'):
        print(f"  Split '{split}': {len(split_class_entries[split])} / 112 classes found.", flush=True)

    # 3. Create or Open HDF5 dataset structure (Resumable)
    num_classes = len(TARGET_112)
    n_testplus = args.n_test + 200
    splits_config = {
        'train': args.n_train * num_classes,
        'val': args.n_val * num_classes,
        'test': args.n_test * num_classes,
        'testplus': n_testplus * num_classes,
    }

    file_mode = "r+" if output_h5.exists() else "w"
    print(f"\nOpening HDF5 container in mode '{file_mode}'...", flush=True)
    h5_file = h5py.File(output_h5, file_mode)
    
    if "categories" not in h5_file:
        h5_file.create_dataset("categories", data=[c.encode('utf-8') for c in TARGET_112])

    for split, n_imgs in splits_config.items():
        if split not in h5_file:
            grp = h5_file.create_group(split)
            grp.create_dataset("data", shape=(n_imgs, args.image_size, args.image_size, 3), dtype=np.uint8)
            grp.create_dataset("labels", shape=(n_imgs,), dtype=np.int32, fillvalue=-1)
            grp.create_dataset("image_size_hwc", data=np.tile([args.image_size, args.image_size, 3], (n_imgs, 1)).astype(np.uint8))
            grp.create_dataset("completed", shape=(n_imgs,), dtype=np.bool_, fillvalue=False)
        else:
            grp = h5_file[split]
            if "completed" not in grp and split in ('train', 'val', 'test'):
                # Retrofit completed mask from existing data
                print(f"  Checking existing progress for split '{split}'...", flush=True)
                labels = grp["labels"][:]
                # If labels contain non-negative numbers and data is non-zero, mark completed
                comp_mask = np.zeros(n_imgs, dtype=np.bool_)
                if np.any(labels >= 0):
                    data_sample = grp["data"][:min(100, n_imgs)]
                    if np.any(data_sample != 0):
                        for i in range(n_imgs):
                            if labels[i] >= 0:
                                comp_mask[i] = True
                grp.create_dataset("completed", data=comp_mask)

    h5_file.flush()

    # 4. Download images in bounded batches per split (Resumable)
    def process_item(item):
        s, g_idx, l_idx, z = item
        img_arr = fetch_and_decode_image(URL, z, args.image_size)
        return s, g_idx, l_idx, img_arr

    for split in ['val', 'test', 'train']:
        print(f"\n--- Processing split: {split.upper()} ---", flush=True)
        n_per_cls = args.n_val if split == 'val' else (args.n_test if split == 'test' else args.n_train)
        
        # Check existing completion mask
        comp_mask = h5_file[split]["completed"][:]
        total_in_split = len(comp_mask)
        num_already_done = int(np.sum(comp_mask))

        if num_already_done >= total_in_split:
            print(f"Split '{split.upper()}' is already 100% complete ({num_already_done}/{total_in_split} images). Skipping!", flush=True)
            continue
        elif num_already_done > 0:
            print(f"Resuming '{split.upper()}': {num_already_done}/{total_in_split} images already downloaded.", flush=True)

        tasks = []
        for cls_idx, cat in enumerate(TARGET_112):
            entries = split_class_entries[split].get(cat, [])
            selected_entries = entries[:n_per_cls]
            for in_cls_idx, zinfo in enumerate(selected_entries):
                global_idx = cls_idx * n_per_cls + in_cls_idx
                if not comp_mask[global_idx]:
                    tasks.append((split, global_idx, cls_idx, zinfo))

        print(f"Remaining to download for '{split.upper()}': {len(tasks)} images.", flush=True)
        pbar = tqdm(total=len(tasks), desc=f"Extracting {split}")

        # Stream in batches of args.batch_size
        batch_size = args.batch_size
        completed_since_start = 0

        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start : batch_start + batch_size]
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_item, t) for t in batch_tasks]
                for fut in as_completed(futures):
                    s, g_idx, l_idx, img_arr = fut.result()
                    h5_file[s]["data"][g_idx] = img_arr
                    h5_file[s]["labels"][g_idx] = l_idx
                    h5_file[s]["completed"][g_idx] = True
                    pbar.update(1)
                    completed_since_start += 1
                    
            # Flush batch to disk and clear heap/GC
            h5_file.flush()
            gc.collect()
            
            total_now = num_already_done + completed_since_start
            print(f"[{split.upper()}] Flushed batch to disk: {total_now}/{total_in_split} ({total_now/total_in_split*100:.1f}%)", flush=True)

        pbar.close()
        h5_file.flush()

    # 5. Populate testplus from test (50) and train (200)
    print("\n--- Populating split: TESTPLUS ---", flush=True)
    for cls_idx in range(num_classes):
        test_start = cls_idx * args.n_test
        tp_start = cls_idx * n_testplus
        # Copy test images
        h5_file["testplus"]["data"][tp_start:tp_start + args.n_test] = h5_file["test"]["data"][test_start:test_start + args.n_test]
        h5_file["testplus"]["labels"][tp_start:tp_start + args.n_test] = cls_idx
        # Copy 200 train images
        train_start = cls_idx * args.n_train
        h5_file["testplus"]["data"][tp_start + args.n_test:tp_start + n_testplus] = h5_file["train"]["data"][train_start:train_start + 200]
        h5_file["testplus"]["labels"][tp_start + args.n_test:tp_start + n_testplus] = cls_idx

    # 6. Compute mean and std channels (streaming chunks to avoid memory errors)
    print("\nCalculating train channel statistics...", flush=True)
    train_data = h5_file['train']['data']
    n_imgs = train_data.shape[0]
    total_pixels = n_imgs * args.image_size * args.image_size
    sum_ch = np.zeros(3, dtype=np.float64)
    sum_sq_ch = np.zeros(3, dtype=np.float64)
    
    chunk_size = 2000
    for start in range(0, n_imgs, chunk_size):
        end = min(start + chunk_size, n_imgs)
        chunk = train_data[start:end].astype(np.float64)
        sum_ch += np.sum(chunk, axis=(0, 1, 2))
        sum_sq_ch += np.sum(chunk**2, axis=(0, 1, 2))
        
    mean_ch = sum_ch / total_pixels
    var_ch = (sum_sq_ch / total_pixels) - (mean_ch ** 2)
    std_ch = np.sqrt(var_ch)
    if "train_img_mean_channels" in h5_file:
        del h5_file["train_img_mean_channels"]
    if "train_img_std_channels" in h5_file:
        del h5_file["train_img_std_channels"]
    h5_file.create_dataset("train_img_mean_channels", data=mean_ch)
    h5_file.create_dataset("train_img_std_channels", data=std_ch)

    h5_file.close()
    print(f"\n✅ Done! Successfully created {output_h5}!", flush=True)
    print(f"Final file size: {output_h5.stat().st_size / (1024**3):.2f} GB", flush=True)


if __name__ == "__main__":
    main()
