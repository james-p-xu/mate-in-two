"""
Download Lichess puzzles, extract Mate-in-2 rows, sort by rating, and save.
"""

import argparse
import requests
from pathlib import Path
import zstandard as zstd
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR  = Path(__file__).resolve().parent 
RAW_ZST  = DATA_DIR / "lichess_db_puzzle.csv.zst"
RAW_CSV  = DATA_DIR / "lichess_puzzles_raw.csv"
OUT_CSV  = DATA_DIR / "mate_in_2_fen_by_rating.csv"

LICHESS_DUMP_URL   = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
COLS       = ["FEN", "Rating", "Themes"]
DOWNLOAD_CHUNK_SIZE = 1 << 20 # 1mb

def download(url: str, dst: Path) -> None:
    """Stream-download with progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B",
                                        unit_scale=True, desc="Downloading") as bar:
            for chunk in r.iter_content(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

def decompress_zst(src: Path, dst: Path) -> None:
    """Decompress .zst → .csv with progress bar."""
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as fin, open(dst, "wb") as fout, \
         tqdm(unit="B", unit_scale=True, desc="Decompressing") as bar:
        reader = dctx.stream_reader(fin)
        while True:
            chunk = reader.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            fout.write(chunk)
            bar.update(len(chunk))

def build_mate1(src_csv: Path, out_csv: Path) -> None:
    """Filter Mate-in-1 puzzles and sort by rating."""
    print("[*] Loading CSV …")
    df = pd.read_csv(src_csv, usecols=COLS,
                     dtype={"FEN": "string", "Rating": "Int32", "Themes": "string"})

    mate2 = df[df["Themes"].str.contains("mateIn1")].copy()
    print(f"[+] Found {len(mate2):,} mate-in-1 puzzles")

    mate2.sort_values("Rating")[["FEN", "Rating"]].to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv}")

def build_mate2(src_csv: Path, out_csv: Path) -> None:
    """Filter Mate-in-2 puzzles and sort by rating."""
    print("[*] Loading CSV …")
    df = pd.read_csv(src_csv, usecols=COLS,
                     dtype={"FEN": "string", "Rating": "Int32", "Themes": "string"})

    mate2 = df[df["Themes"].str.contains("mateIn2")].copy()
    print(f"[+] Found {len(mate2):,} mate-in-2 puzzles")

    mate2.sort_values("Rating")[["FEN", "Rating"]].to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv}")

def remove(path: Path):
    try:
        path.unlink()
        print(f"[-] Deleted {path}")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Mate-in-N dataset from Lichess dump.")
    parser.add_argument("--theme", required=True,
                        choices=["m1", "m2"],
                        help="Puzzle theme to filter (mate-in-1 or mate-in-2)")
    parser.add_argument("--out", required=True,
                        help="Output CSV file path (including filename)")
    args = parser.parse_args()

    # download and decompress
    download(LICHESS_DUMP_URL, RAW_ZST)
    decompress_zst(RAW_ZST, RAW_CSV)

    # build dataset based on theme
    out_csv = Path(args.out)
    if args.theme == "mateIn1":
        build_mate1(RAW_CSV, out_csv)
    else:
        build_mate2(RAW_CSV, out_csv)

    # clean up intermediate artifacts
    remove(RAW_ZST)
    remove(RAW_CSV)
