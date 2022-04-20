import os, sys, glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

# my utils
from utils import xmatch
from utils import query_photoz_datalab as photoz

"""
Cross-matching a post-run list of candidates with:
- GAIA (return distance to closest match within 5", parallax, mags): done 2''
- Simbad (return distance to closest match and classification?): done 5''
- Legacy Survey (return what Igor's code queries for closest match): done
- TNS entries (is this possible?): done from January 2021
- AGN WISE colors: done xmatch, TODO AGN space
- TESS (return distance to closest match within 5", mags, "class"): done
- SkyMapper (return distance to closest match within 5", mags, "class"): done
# - Variable star catalogs? (return distance to closest match within 5" and if it's a known var star?)
"""

if __name__ == "__main__":
    """Process light-curves with Fink inspired features & xmatches
    https://github.com/astrolabsoftware/fink-filters
    """
    parser = argparse.ArgumentParser(description="Compute candidate features + xmatch")

    parser.add_argument(
        "--path_in",
        type=str,
        default="../Candidates_postrunsummary.csv",
        help="Path+fname input (ra,dec) ",
    )
    parser.add_argument(
        "--path_out", type=str, default="./dump_postrun", help="Path to outputs",
    )
    parser.add_argument(
        "--path_tns",
        type=str,
        default="../tns_search-202101-0420_ra48-94_dec-70-4.csv",
        help="Path to TNS dump to cross-match",
    )
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    # read input file
    df = pd.read_csv(args.path_in)

    # convert to degrees
    coo = SkyCoord(
        df["ra (hms)"].astype(str),
        df["dec (dms)"].astype(str),
        unit=(u.hourangle, u.deg),
    )
    df["ra"] = coo.ra.degree
    df["dec"] = coo.dec.degree
    df["ObjectId"] = np.arange(0, len(df))

    print("SIMBAD xmatch")
    z, sptype, typ, ctlg = xmatch.cross_match_simbad(
        df["ObjectId"].to_list(), df["ra"].to_list(), df["dec"].to_list()
    )
    # save in df
    df["simbad_type"] = typ
    df["simbad_ctlg"] = ctlg
    df["simbad_sptype"] = sptype
    df["simbad_redshift"] = z

    print("GAIA2 + e3 xmatch")
    source, ragaia, decgaia, plx, plxerr, gmag, angdist = xmatch.cross_match_gaia(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:I/345/gaia2",
    )
    (
        source_edr3,
        ragaia_edr3,
        decgaia_edr3,
        plx_edr3,
        plxerr_edr3,
        gmag_edr3,
        angdist_edr3,
    ) = xmatch.cross_match_gaia(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:I/350/gaiaedr3",
    )
    # save in df
    df["gaia_DR2_source"] = source
    df["gaia_DR2_ra"] = ragaia
    df["gaia_DR2_dec"] = decgaia
    df["gaia_DR2_parallax"] = plx
    df["gaia_DR2_parallaxerr"] = plxerr
    df["gaia_DR2_gmag"] = gmag
    df["gaia_DR2_angdist"] = angdist
    df["gaia_eDR3_source"] = source_edr3
    df["gaia_eDR3_ra"] = ragaia_edr3
    df["gaia_eDR3_dec"] = decgaia_edr3
    df["gaia_eDR3_parallax"] = plx_edr3
    df["gaia_eDR3_parallaxerr"] = plxerr_edr3
    df["gaia_eDR3_gmag"] = gmag_edr3
    df["gaia_eDR3_angdist"] = angdist_edr3

    print("USNO-A.20 xmatch")
    (source_usno, angdist_usno,) = xmatch.cross_match_usno(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:I/252/out",
    )
    df["USNO_source"] = source_usno
    df["USNO_angdist"] = angdist_usno

    print("Legacy Survey xmatch")
    list_ls_df = []
    for (idx, ra, dec) in df[["ObjectId", "ra", "dec"]].values:
        list_ls_df.append(photoz.query_coords_ls(idx, ra, dec, radius_arcsec=10))
    df_ls = pd.concat(list_ls_df)
    df_ls = df_ls.rename(columns={"id": "ObjectId"})
    print("Finished Legacy Survey xmatch")
    df = pd.merge(df, df_ls, on="ObjectId")

    print("WISE xmatch")
    df_wise = xmatch.cross_match_alerts_raw_generic(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:II/311/wise",
        distmaxarcsec=5,
    )
    df_wise = df_wise.rename(
        columns={
            col: col + "_wise"
            for col in df_wise.columns
            if col not in ["ObjectId", "ra", "dec"]
        }
    )
    df_wise["ObjectId"] = df_wise["ObjectId"].astype(int)
    df_wise["ra"] = df_wise["ra"].astype(float)
    df_wise["dec"] = df_wise["dec"].astype(float)
    df = pd.merge(df, df_wise, on=["ObjectId", "ra", "dec"], how="left")

    print("SkyMapper xmatch")
    df_smss = xmatch.cross_match_alerts_raw_generic(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:II/358/smss",
        distmaxarcsec=5,
    )
    df_smss = df_smss.rename(
        columns={
            col: col + "_SMSS"
            for col in df_smss.columns
            if col not in ["ObjectId", "ra", "dec"]
        }
    )
    df_smss["ObjectId"] = df_smss["ObjectId"].astype(int)
    df_smss["ra"] = df_smss["ra"].astype(float)
    df_smss["dec"] = df_smss["dec"].astype(float)
    df = pd.merge(df, df_smss, on=["ObjectId", "ra", "dec"], how="left")

    print("TESS xmatch")
    df_tess = xmatch.cross_match_alerts_raw_generic(
        df["ObjectId"].to_list(),
        df["ra"].to_list(),
        df["dec"].to_list(),
        ctlg="vizier:IV/39/tic82",
        distmaxarcsec=5,
    )
    df_tess = df_tess.rename(
        columns={
            col: col + "_TESS"
            for col in df_tess.columns
            if col not in ["ObjectId", "ra", "dec"]
        }
    )
    df_tess["ObjectId"] = df_tess["ObjectId"].astype(int)
    df_tess["ra"] = df_tess["ra"].astype(float)
    df_tess["dec"] = df_tess["dec"].astype(float)
    df = pd.merge(df, df_tess, on=["ObjectId", "ra", "dec"], how="left")

    print(f"TNS dump xmatch {args.path_tns}")
    df_tns = pd.read_csv(args.path_tns)
    df_tns = df_tns.rename(columns={"RA": "ra (h)", "DEC": "dec (h)"})
    # convert to degrees
    coo_tns = SkyCoord(
        ra=df_tns["ra (h)"].astype(str),
        dec=df_tns["dec (h)"].astype(str),
        unit=(u.hourangle, u.deg),
    )
    # cross-match
    idx, d2d, d3d = coo.match_to_catalog_sky(coo_tns, nthneighbor=1)

    # set separation length
    sep_constraint = d2d.arcmin < 5
    catalog_matches = np.unique(df["ObjectId"].values[idx[sep_constraint]])

    # identify position of matches in the input dataframe
    pdf_matches = pd.DataFrame(
        {"ObjectId": np.array(catalog_matches, dtype=np.int64), "match_TNS": True}
    )
    df = pd.merge(df, pdf_matches, how="left", on="ObjectId")

    m = df["match_TNS"].apply(lambda x: x is True)
    # Now get types for these
    catalog_ztf_merge = SkyCoord(
        ra=np.array(df.loc[m, "ra"].values, dtype=np.float) * u.degree,
        dec=np.array(df.loc[m, "dec"].values, dtype=np.float) * u.degree,
    )

    # cross-match
    idx2, d2d2, d3d2 = catalog_ztf_merge.match_to_catalog_sky(coo_tns)

    df["TNS Name"] = "Unknown"
    df.loc[m, "TNS Name"] = [
        str(i).strip() for i in df_tns["Name"].astype(str).values[idx2]
    ]

    df.to_csv(f"{args.path_out}/{Path(args.path_in).stem}_xmatched.csv")
