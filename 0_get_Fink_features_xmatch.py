# Year 2022
# Authors: Anais Möller based on fink-broker.org code

import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# my utils
from utils import xmatch
from utils import mag_color


def setup_logging(logname):
    logger = None

    # Create logger using python logging module
    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.DEBUG)

    logging_handler_err = logging.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(logging.WARNING)

    logger = logging.getLogger("localLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging_handler_out)
    logger.addHandler(logging_handler_err)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"logs/{logname}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def process_fn(inputs):
    fn, fil = inputs
    return fn(fil)


def process_single_file(fname):
    # read file and convert to pandas
    try:
        df_tmp = Table.read(fname, format="ascii").to_pandas()
        if set(["ra", "dec"]).issubset(df_tmp.keys()):
            # get id
            idx = Path(fname).stem.replace(".forced.difflc", "")

            # get ra,dec, idx for xmatch
            ra_tmp, dec_tmp = df_tmp["ra"][0], df_tmp["dec"][0]
            # convert to degrees
            coo = SkyCoord(ra_tmp, dec_tmp, unit=(u.hourangle, u.deg))
            out_ra = coo.ra.degree
            out_dec = coo.dec.degree

            # get color, dmag and rate
            (
                dmag_i,
                dmag_r,
                dmag_rate_r,
                dmag_rate_i,
                color,
                color_avg,
                max_mag_i,
                max_mag_r,
            ) = mag_color.last_color_rate(df_tmp)

            ndet = len(df_tmp)
            # clean
            del df_tmp

            df_out = pd.DataFrame()
            df_out["id"] = [idx]
            df_out["ra"] = [out_ra]
            df_out["dec"] = [out_dec]
            df_out["max_mag_i"] = [max_mag_i]
            df_out["max_mag_"] = [max_mag_r]
            df_out["dmag_i"] = [dmag_i]
            df_out["dmag_r"] = [dmag_r]
            df_out["dmag_rate_i"] = [dmag_rate_i]
            df_out["dmag_rate_r"] = [dmag_rate_r]
            df_out["color"] = [color]
            df_out["color_avg"] = [color_avg]
            df_out["ndet"] = [ndet]

        else:
            df_out = pd.DataFrame()
    except Exception:
        print("File corrupted or not readable", fname)
        df_out = pd.DataFrame()
    return df_out


if __name__ == "__main__":
    """Process light-curves with Fink inspired features & xmatches
    https://github.com/astrolabsoftware/fink-filters
    """
    parser = argparse.ArgumentParser(description="Compute candidate features + xmatch")

    parser.add_argument(
        "--path_field",
        type=str,
        default="./S82sub8_tmpl",
        help="Path to field",
    )
    parser.add_argument(
        "--run",
        type=int,
        default="12",
        help="Run number (int next to field/ccd)",
    )
    parser.add_argument(
        "--path_out",
        type=str,
        default="./Fink_outputs",
        help="Path to outputs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug: loop processing (slow)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="one file processed only",
    )
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)
    os.makedirs("logs/", exist_ok=True)

    logname = f"{Path(args.path_field).stem}_{args.run}_preprocess"
    logger = setup_logging(logname)

    # read files
    list_files = glob.glob(f"{args.path_field}/*/*{args.run}/*.forced.difflc.txt")
    print(f"{len(list_files)} files found in {args.path_field} for run {args.run}")

    if args.test:
        print(list_files)
        print("Processing only one file", list_files[0])
        df = process_single_file(list_files[0])
    elif args.debug:
        print(list_files)
        # no parallel
        list_proc = []
        for fil in list_files:
            logger.info(fil)
            list_proc.append(process_single_file(fil))
        df = pd.concat(list_proc)
    else:
        # Read and process files faster with ProcessPoolExecutor
        max_workers = multiprocessing.cpu_count()
        # use parallelization to speed up processing
        # Split list files in chunks of size 10 or less
        # to get a progress bar and alleviate memory constraints
        num_elem = len(list_files)
        num_chunks = num_elem // 10 + 1
        list_chunks = np.array_split(np.arange(num_elem), num_chunks)

        process_fn_file = partial(process_single_file)

        list_fn = []
        for fmt in list_files:
            list_fn.append(process_fn_file)
        list_processed = []
        for chunk_idx in tqdm(list_chunks, desc="Process", ncols=100):
            # Process each file in the chunk in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                start, end = chunk_idx[0], chunk_idx[-1] + 1
                # Need to cast to list because executor returns an iterator
                list_pairs = list(zip(list_fn[start:end], list_files[start:end]))
                list_processed += list(executor.map(process_fn, list_pairs))

        df = pd.concat(list_processed)
    # x match
    z, sptype, typ, ctlg = xmatch.cross_match_simbad(
        df["id"].to_list(), df["ra"].to_list(), df["dec"].to_list()
    )

    # save in df
    df["simbad_type"] = typ
    df["simbad_ctlg"] = ctlg
    df["simbad_sptype"] = sptype
    df["simbad_redshift"] = z

    outname = str(Path(args.path_field).stem)
    df.to_csv(f"{args.path_out}/{outname}_{args.run}.csv", index=False)
