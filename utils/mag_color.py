# Year 2022
# Authors: Anais Möller based on fink-broker.org code

import numpy as np
import pandas as pd
from astropy.time import Time

# ["g", "r", "i", "z"], value=[1, 2, 3, 4]
dic_filters = {1: "g", 2: "r", 3: "i", 4: "z"}
pd.options.mode.chained_assignment = None


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
    """ Extract last i-g and delta mag for each object
    
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
    dic_dmag = {1: [], 2: [], 3: [], 4: []}
    dic_dmag_mjd = {1: [], 2: [], 3: [], 4: []}
    dic_rate = {1: [], 2: [], 3: [], 4: []}
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
    pdf["mjd_int"] = pdf["mjd"].astype(int).values
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


def last_color_rate(df_tmp):
    """ Compute color and rate for a light-curve

    Return:
    - dmag_i : last delta magnitude in i 
    """

    # If daily cadence then = rate
    smooth_by = 1

    # eliminate empty mags
    df_tmp = df_tmp[df_tmp["m"] != "-"]
    df_tmp["magnitude"] = df_tmp["m"].astype(np.float).copy()

    # variable reformatting
    t = Time(df_tmp["dateobs"].to_list(), format="isot", scale="utc")
    df_tmp["mjd"] = t.mjd.astype(float).copy()
    df_tmp["filter"] = (
        df_tmp["filt"]
        .replace(to_replace=["g", "r", "i", "z"], value=[1, 2, 3, 4])
        .copy()
    )

    dic_dmag, dic_dmag_mjd, dic_rate, color, color_mjd = extract_delta_color(
        df_tmp, smooth_by=smooth_by
    )

    dmag_i = dic_dmag[2][-1] if len(dic_dmag[2]) > 0 else np.nan
    dmag_r = dic_dmag[3][-1] if len(dic_dmag[3]) > 0 else np.nan
    dmag_rate_i = dic_rate[2][-1] if len(dic_rate[2]) > 0 else np.nan
    dmag_rate_r = dic_rate[3][-1] if len(dic_rate[3]) > 0 else np.nan
    last_color = color[-1] if len(color) > 0 else np.nan
    color_avg = np.array(color).mean() if len(color) > 0 else np.nan

    return dmag_i, dmag_r, dmag_rate_r, dmag_rate_i, last_color, color_avg
