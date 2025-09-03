from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os, hashlib, string
from collections import deque

# ---------------- Config ----------------
N_GRAM_SIZE = 5
MAX_WORKERS  = os.cpu_count() or 1
BYTES_PER_CHUNK = 16 * 1024 * 1024
OVERLAP_BYTES   = 4096

# --------- Tokenization / Hash ----------
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
def fast_tokenize(s: str): return s.translate(_PUNCT_TABLE).lower().split()

def stable_ngram_hash(ngram_tuple):
    data = "\x1f".join(ngram_tuple).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

# ------------- Chunking -----------------
def chunk_ranges(filesize, size=BYTES_PER_CHUNK):
    starts = list(range(0, filesize, size))
    return [(s, min(s+size, filesize)) for s in starts]

def overlap_tokens_before(file_path: Path, start: int) -> list[str]:
    if start == 0: return []
    beg = max(0, start - OVERLAP_BYTES)
    with open(file_path, "rb") as f:
        f.seek(beg); buf = f.read(start - beg)
    # only keep text up to the boundary
    toks = fast_tokenize(buf.decode("utf-8", errors="ignore"))
    return toks[-(N_GRAM_SIZE-1):]

# ------------- Worker -------------------
def process_range(args):
    file_path, start, end, overlap_prefix = args
    ngrams = set()
    window = deque(overlap_prefix[-(N_GRAM_SIZE-1):], maxlen=N_GRAM_SIZE)
    warmup = len(window)

    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

    for tok in fast_tokenize(data):
        window.append(tok)
        if len(window) == N_GRAM_SIZE:
            if warmup > 0:
                warmup -= 1  # skip windows that start before 'start'
            else:
                ngrams.add(stable_ngram_hash(tuple(window)))
    return ngrams

# ------------- Per-file build -----------
def build_file_ngrams_parallel(path: Path) -> set[int]:
    size = os.path.getsize(path)
    ranges = chunk_ranges(size)
    tasks = []
    for (s, e) in ranges:
        tasks.append((str(path), s, e, overlap_tokens_before(path, s)))
    nset = set()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for part in ex.map(process_range, tasks, chunksize=1):
            nset.update(part)
    return nset

# ------------- Containment --------------
def calculate_containment(A: set, B: set) -> float:
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    smaller, larger = (A, B) if len(A) <= len(B) else (B, A)
    inter = sum(1 for x in smaller if x in larger)
    return inter / len(smaller)

# ------------- Main ---------------------
if __name__ == "__main__":
    data_dir = Path("./containment_data_test")
    files = list(data_dir.glob("*.txt"))
    fnames = [f.stem for f in files]
    fname_to_path = {f.stem: f for f in files}
    print(f"Files: {fnames}")

    file_name_ngram_sets = {}
    for f in tqdm(files, desc="Building n-grams (per-file parallel)"):
        file_name_ngram_sets[f.stem] = build_file_ngrams_parallel(f)

    containment_matrix = pd.DataFrame(index=fnames, columns=fnames, dtype=float)
    print("\nCalculating pairwise containment...")
    total_pairs = len(fnames) * (len(fnames) - 1) // 2
    with tqdm(total=total_pairs, desc="Computing containment matrix") as pbar:
        for i, fi in enumerate(fnames):
            for j, fj in enumerate(fnames):
                if i == j:
                    containment_matrix.loc[fi, fj] = 1.0
                elif i < j:
                    c = calculate_containment(file_name_ngram_sets[fi], file_name_ngram_sets[fj])
                    containment_matrix.loc[fi, fj] = c
                    containment_matrix.loc[fj, fi] = c
                    pbar.update(1)

    containment_matrix.to_csv("containment_matrix.csv")
    print("\nMatrix saved to: containment_matrix.csv")

    # Pretty print (optional)
    file_sizes = {k: len(v) for k, v in file_name_ngram_sets.items()}
    sorted_fnames = [k for k, _ in sorted(file_sizes.items(), key=lambda kv: kv[1])]
    print("\nFile".ljust(15) + "".join(f"{n[:8]:>10}" for n in sorted_fnames))
    for i, r in enumerate(sorted_fnames):
        row = f"{r[:14]:15}" + " " * (10 * i)
        for j in range(i, len(sorted_fnames)):
            v = containment_matrix.loc[r, sorted_fnames[j]]
            row += f"{'1.000':>10}" if i == j else f"{v*100:>9.1f}%"
        print(row)
