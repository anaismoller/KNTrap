import io
import csv
import logging
import requests
import glob, os
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# ["g", "r", "i", "z"], value=[1, 2, 3, 4]
dic_filters = {1: "g", 2: "r", 3: "i", 4: "z"}


def i_minus_g(fid, mag):
    """ Compute i-g based on vectors of filters and magnitudes
    """

    if list(set(fid)) == [1, 3]:
        # last measurement
        last_fid = fid[-1]

        # last measurement with different filter
        # could take the mean
        index_other = np.where(np.array(fid) != last_fid)[0][-1]

        sign = np.diff([fid[index_other], last_fid])[0]
        mag = [mag[index_other], mag[-1]]

        return -1 * sign * np.diff(mag)[0]

    else:
        return np.nan


def extract_delta_color(pdf: pd.DataFrame, smooth_by: float):
    """ Extract last g-r and delta mag for each object
    
    Modified from Fink https://github.com/astrolabsoftware/fink-science-portal/apps/utils.py
    Adding a grouping for a given mjd interval
    
    Parameters
    ----------
    pdf: pandas DataFrame
        DataFrame containing magnitude, mjd, filters
        Filter band, as integer. g: 1, r: 2, i: 3, z:4 
    smooth_by: float
        MJD delta to smooth computation

    Returns
    ----------

    
    """

    # dmag, rate for each filter
    dic_dmag = dict.fromkeys([1, 2, 3, 4])
    dic_dmag_mjd = dict.fromkeys([1, 2, 3, 4])
    dic_rate = dict.fromkeys([1, 2, 3, 4])
    for fil in pdf["filter"].unique():
        subpdf = pdf[pdf["filter"] == fil]
        subpdf = subpdf.sort_values("mjd", ascending=False)

        # dmag smoothed
        min_mjd = float(int(subpdf.mjd.min()))
        max_mjd = float(int(subpdf.mjd.max()))
        bins = np.arange(min_mjd, max_mjd + 1, smooth_by)
        df_grouped = subpdf.groupby(np.digitize(subpdf.mjd, bins)).median()
        mag_grouped = df_grouped["magnitude"]
        mjd_grouped = df_grouped["mjd"]
        if len(mag_grouped) > 1:
            # only compute if more than one observation
            dmag_ = mag_grouped.diff(periods=1).values[1:]
            dmag_mjd_ = mjd_grouped.values[1:].astype(int)
            dic_dmag[fil] = dmag_
            dic_dmag_mjd[fil] = dmag_mjd_

        # Rate by day
        bins = np.arange(min_mjd, max_mjd + 1, 1)
        df_grouped_byday = subpdf.groupby(np.digitize(subpdf.mjd, bins)).median()
        mag_grouped_byday = df_grouped_byday["magnitude"]
        rate_ = mag_grouped_byday.diff(periods=1).values
        if len(mag_grouped_byday) > 1:
            dic_rate[fil] = rate_[1:]

    # for color (can be modified to take median mag)
    # group by night
    gpdf = pdf.groupby("mjd_int")[["filter", "mjd", "magnitude"]].agg(list)
    # take only nights with at least measurements on 2 different filters
    mask = gpdf["filter"].apply(lambda x: (len(x) > 1) & (np.sum(x) / len(x) != x[0]))
    gpdf_night = gpdf[mask]

    # compute i-g for those nights
    color_tmp = [
        i_minus_g(i, j)
        for i, j in zip(gpdf_night["filter"].values, gpdf_night["magnitude"].values)
    ]
    mask = np.argwhere(~np.isnan(color_tmp)).flatten()
    if len(mask) > 1:
        color_mjd = gpdf_night.index.values[mask]
        color = [color_tmp[k] for k in mask]
    else:
        color_mjd = []
        color = []

    return (dic_dmag, dic_dmag_mjd, dic_rate, color, color_mjd)


if __name__ == "__main__":
    inpath = "./S82sub8_59.12"
    list_files = glob.glob(f"{inpath}/*.forced.difflc.txt")

    # If daily cadence then = rate
    smooth_by = 1

    for fname in list_files[:1]:
        # read file and convert to pandas
        df_tmp = Table.read(fname, format="ascii").to_pandas()

        # variable reformatting
        t = Time(df_tmp["dateobs"].to_list(), format="isot", scale="utc")
        df_tmp["mjd"] = t.mjd.astype(float)
        df_tmp["mjd_int"] = t.mjd.astype(int)
        df_tmp["magnitude"] = df_tmp["m"].values
        df_tmp["filter"] = df_tmp["filt"].replace(
            to_replace=["g", "r", "i", "z"], value=[1, 2, 3, 4]
        )
        df_tmp = df_tmp[df_tmp["magnitude"] != "-"]
        df_tmp["magnitude"] = df_tmp["magnitude"].astype(np.float).copy()

        dic_dmag, dic_dmag_mjd, dic_rate, color, color_mjd = extract_delta_color(
            df_tmp, smooth_by=smooth_by
        )

        import ipdb

        ipdb.set_trace()

        # # Plot to verify processing
        # from matplotlib import gridspec

        # fig = plt.figure(figsize=(14, 14))
        # gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        # ax0 = plt.subplot(gs[0, 0])
        # ax1 = plt.subplot(gs[1, 0])
        # x_arr = np.arange(df_tmp.mjd.min(), df_tmp.mjd.max() + 1, 1)
        # ax1.plot(x_arr, np.zeros(len(x_arr)), color="grey", linestyle="-")

        # for fil in df_tmp["filter"].unique():
        #     sel = df_tmp[df_tmp["filter"] == fil]
        #     if len(sel) > 1:
        #         ax0.errorbar(
        #             sel.mjd,
        #             sel.magnitude,
        #             yerr=sel.dm.astype(float),
        #             fmt="o",
        #             label=dic_filters[fil],
        #         )
        #         if fil in dic_dmag_mjd.keys():
        #             ax1.scatter(dic_dmag_mjd[fil], dic_dmag[fil])
        #             if len(color) > 0:
        #                 ax1.scatter(color_mjd, color, color="black")
        # idx = Path(fname).stem.replace(".forced.difflc", "")
        # ax0.legend()
        # plt.savefig(f"{idx}.png")
