# Year 2022
# Authors: Anais Möller based on fink-broker.org code

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

list_simbad_galaxies = [
    "galaxy",
    "Galaxy",
    "EmG",
    "Seyfert",
    "Seyfert_1",
    "Seyfert_2",
    "BlueCompG",
    "StarburstG",
    "LSB_G",
    "HII_G",
    "High_z_G",
    "GinPair",
    "GinGroup",
    "BClG",
    "GinCl",
    "PartofG",
]
keep_cds = ["Unknown", "Transient", "Fail"] + list_simbad_galaxies


if __name__ == "__main__":
    """Filter interesting KN candidates
    Inspired from Fink filter
    https://github.com/astrolabsoftware/fink-filters/blob/master/fink_filters/filter_kn_candidates/filter.py
    """
    parser = argparse.ArgumentParser(description="Compute candidate features + xmatch")

    parser.add_argument(
        "--path_out",
        type=str,
        default="./Fink_outputs",
        help="Path to outputs",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="/S82sub8_59.12.csv",
        help="Filename of features",
    )
    args = parser.parse_args()

    print(f"Filtering {args.fname}")
    df = pd.read_csv(f"{args.path_out}/{args.fname}")

    # keep only candidates that are unknown transients/close-by galaxy
    cut_simbad = df.simbad_type.isin(keep_cds)
    cut_rate = (df.dmag_rate_r > 0.3) | (
        df.dmag_rate_i > 0.3
    )  # Andreoni et al. 2021 https://arxiv.org/abs/2104.06352
    # cut_new_det = df.ndet < 4  # number of detections
    # df_sel = df[cut_simbad & cut_rate & cut_new_det]
    df_sel = df[cut_simbad & cut_rate]

    # add date tag
    # tt = datetime.now()
    # date_to_print = tt.strftime("%Y%m%d")
    out_prefix = Path(args.fname).stem
    df_sel.to_csv(f"{args.path_out}/selected_{out_prefix}.csv", index=False)
    print(f"Selected {len(df_sel)} from {len(df)}")
