import os
import shlex
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from itertools import product

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrapper for selection")

    parser.add_argument(
        "--path_field", type=str, default="./S82sub8_tmpl", help="Path to field",
    )
    parser.add_argument(
        "--run", type=int, default="12", help="Run number (int next to field/ccd)",
    )
    parser.add_argument(
        "--path_out", type=str, default="./Fink_outputs", help="Path to outputs",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug: one file processed only",
    )

    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    cmd = (
        "python -W ignore 0_get_Fink_features_xmatch.py "
        f"--path_field {args.path_field} --path_out {args.path_out} --run {args.run} "
    )
    if args.debug:
        cmd += "--debug"
    subprocess.check_call(shlex.split(cmd))

    fname = args.path_field.split("/")[-1]
    print(f"Features from {fname}")
    cmd = (
        "python -W ignore 1_filter_top_candidates.py "
        f"--fname {fname}_{args.run}.csv --path_out {args.path_out} "
    )
    subprocess.check_call(shlex.split(cmd))
