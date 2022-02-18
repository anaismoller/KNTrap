import os
import shlex
import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrapper for selection")

    parser.add_argument(
        "--path_field", type=str, default="./S82sub8_tmpl", help="Path to field",
    )
    parser.add_argument(
        "--path_out", type=str, default="./Fink_outputs", help="Path to outputs",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug: files processed in loop",
    )
    parser.add_argument(
        "--test", action="store_true", help="Debug: one file processed only",
    )

    args = parser.parse_args()
    path_field = args.path_field
    path_out = args.path_out

    os.makedirs(args.path_out, exist_ok=True)

    # 0. process data
    cmd = (
        "python -W ignore 0_get_Fink_features_xmatch.py "
        f"--path_field {path_field} --path_out {path_out} "
    )
    if args.debug:
        cmd += "--debug"
    if args.test:
        cmd += "--test"
    subprocess.check_call(shlex.split(cmd))

    # 1. Filtering
    tmp = Path(path_field).name
    fname = f"{tmp}.csv"
    print(f"Features from {fname}")
    cmd = (
        "python -W ignore 1_filter_top_candidates.py "
        f"--fname {fname} --path_out {path_out} "
    )
    subprocess.check_call(shlex.split(cmd))
