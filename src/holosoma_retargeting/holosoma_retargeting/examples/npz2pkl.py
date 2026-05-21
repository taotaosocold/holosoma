"""
Convert retargeted .npz files to .pkl format.

The .npz file is expected to contain a 'qpos' array and an 'fps' value
(output of the holosoma retargeting pipeline).

qpos layout per frame:
  [0:3]   -> root_pos  (x, y, z)
  [3:7]   -> root_rot  (quaternion w, x, y, z)
  [7:]    -> dof_pos   (joint angles)

Output .pkl dict:
  {
      "fps":      float,
      "root_pos": np.ndarray (N, 3),
      "root_rot": np.ndarray (N, 4),
      "dof_pos":  np.ndarray (N, num_dof),
  }

Usage:
  # Convert a single file
  python npz2pkl.py input.npz

  # Convert a single file to a specific output path
  python npz2pkl.py input.npz -o output.pkl

  # Batch-convert a whole directory (recursive)
  python npz2pkl.py /path/to/npz_dir/

  # Batch-convert with a custom output directory
  python npz2pkl.py /path/to/npz_dir/ -o /path/to/pkl_dir/
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def npz_to_pkl(npz_path: str | Path, pkl_path: str | Path | None = None) -> Path:
    """Convert a single retargeted .npz file to .pkl format.

    Parameters
    ----------
    npz_path : path to the input .npz file.
    pkl_path : path for the output .pkl file.
                If *None*, the .npz extension is replaced with .pkl.

    Returns
    -------
    Path to the written .pkl file.
    """
    npz_path = Path(npz_path)
    if pkl_path is None:
        pkl_path = npz_path.with_suffix(".pkl")
    else:
        pkl_path = Path(pkl_path)

    data = np.load(str(npz_path), allow_pickle=True)

    if "qpos" not in data:
        raise KeyError(f"'qpos' not found in {npz_path}. Available keys: {list(data.keys())}")

    qpos = data["qpos"]  # (N, 7 + num_dof)
    fps = float(data["fps"]) if "fps" in data else 30.0

    root_pos = qpos[:, 0:3].copy()   # (N, 3)
    root_rot = qpos[:, [4, 5, 6, 3]].copy()   # (N, 4)
    dof_pos = qpos[:, 7:].copy()     # (N, num_dof)

    pkl_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)

    return pkl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert retargeted .npz files to .pkl format."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a single .npz file or a directory containing .npz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output path. For a single file, this is the .pkl path. "
            "For a directory, this is the output directory (mirrors input structure)."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # ---------- single file ----------
        if not input_path.suffix == ".npz":
            print(f"Error: {input_path} is not a .npz file.", file=sys.stderr)
            sys.exit(1)
        out = npz_to_pkl(input_path, args.output)
        print(f"Converted: {input_path} -> {out}")

    elif input_path.is_dir():
        # ---------- batch convert ----------
        npz_files = sorted(input_path.rglob("*.npz"))
        if not npz_files:
            print(f"No .npz files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output) if args.output else input_path
        for npz_file in npz_files:
            rel = npz_file.relative_to(input_path)
            pkl_file = output_dir / rel.with_suffix(".pkl")
            out = npz_to_pkl(npz_file, pkl_file)
            print(f"Converted: {npz_file} -> {out}")

        print(f"\nDone. {len(npz_files)} file(s) converted.")

    else:
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
