# Year 2022
# Authors: Anais Möller based on fink-broker.org code

import os
import glob
import argparse
import pandas as pd
from pathlib import Path
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

# my utils
from utils import xmatch
from utils import mag_color


if __name__ == "__main__":
    """ Process light-curves with Fink inspired features & xmatches
    https://github.com/astrolabsoftware/fink-filters
    """
    parser = argparse.ArgumentParser(description="Compute candidate features + xmatch")

    parser.add_argument(
        "--path_field", type=str, default="./S82sub8_59.12", help="Path to field",
    )
    parser.add_argument(
        "--path_out", type=str, default="./Fink_outputs", help="Path to outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    # read files
    list_files = glob.glob(f"{args.path_field}/*.forced.difflc.txt")
    list_idx = []
    list_ra = []
    list_dec = []
    list_dmag_i = []
    list_dmag_r = []
    list_dmag_rate_i = []
    list_dmag_rate_r = []
    list_color = []
    list_color_avg = []
    list_ndet = []
    for fname in list_files:
        # read file and convert to pandas
        df_tmp = Table.read(fname, format="ascii").to_pandas()

        # get id
        list_idx.append(Path(fname).stem.replace(".forced.difflc", ""))

        # get ra,dec, idx for xmatch
        ra_tmp, dec_tmp = df_tmp["ra"][0], df_tmp["dec"][0]
        # convert to degrees
        coo = SkyCoord(ra_tmp, dec_tmp, unit=(u.hourangle, u.deg))
        list_ra.append(coo.ra.degree)
        list_dec.append(coo.dec.degree)

        # get color, dmag and rate
        (
            dmag_i,
            dmag_r,
            dmag_rate_r,
            dmag_rate_i,
            color,
            color_avg,
        ) = mag_color.last_color_rate(df_tmp)

        list_dmag_i.append(dmag_i)
        list_dmag_r.append(dmag_r)
        list_dmag_rate_i.append(dmag_rate_i)
        list_dmag_rate_r.append(dmag_rate_r)
        list_color.append(color)
        list_color_avg.append(color_avg)
        list_ndet.append(len(df_tmp))

        # clean
        del df_tmp

    # x match
    z, sptype, typ, ctlg = xmatch.cross_match_simbad(list_idx, list_ra, list_dec)

    # save in df
    df = pd.DataFrame()
    df["id"] = list_idx
    df["simbad_type"] = typ
    df["simbad_ctlg"] = ctlg
    df["simbad_sptype"] = sptype
    df["simbad_redshift"] = z
    df["dmag_i"] = list_dmag_i
    df["dmag_r"] = list_dmag_r
    df["dmag_rate_i"] = list_dmag_rate_i
    df["dmag_rate_r"] = list_dmag_rate_r
    df["color"] = list_color
    df["color_avg"] = list_color_avg
    df["ndet"] = list_ndet

    df.to_csv(f"{args.path_out}/{Path(list_files[0]).parent}.csv", index=False)

