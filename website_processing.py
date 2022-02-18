import re
import glob
import argparse
import pandas as pd
from bs4 import BeautifulSoup

""" Author Dougal Dobie

    Use metadata (ranked candidates) to reorder YSE pipeline webportal html
"""


def build_ordered_file(filename, metadata):
    heading_format_str = (
        '<center><b><font color="black"><font size=+4>{}</font></font></b>\n'
    )

    # The HTML isn't well-formed, so we have to manually find key lines
    lines = open(filename, "r").readlines()

    start_lines = []
    end_lines = []

    # Find which lines correspond to the start/end of each source
    for i, line in enumerate(lines):
        if (
            line.rstrip()
            == '<table border="0" cellpadding="2" cellspacing="2" style="width:0px">'
        ):
            start_lines.append(i)
        if "<p><strong><em>References:</em></strong><br />" in line:
            end_lines.append(i + 10)

    # Chuck all the junk at the start/end into strings for later
    start_file_str = "".join(lines[: start_lines[0]])
    end_file_str = "".join(lines[end_lines[-1] + 1 :])

    # Build strings for each source
    source_strs = []
    for i, start_line in enumerate(start_lines):
        # if i == len(start_lines)-1:
        #    break
        source_lines = lines[start_lines[i] : end_lines[i] + 1]

        # source_lines = add_robot_scores(source_lines, "0,1,3,4", "0,1,3,4,5")
        # source_str = "".join(source_lines)
        source_strs.append(source_lines)

    # Extract the candidate name and put them in a dict
    # This could easily be combined with the loop above
    source_str_dict = {}
    for source_str in source_strs:
        source_soup = BeautifulSoup("".join(source_str), "html.parser")
        cand_id = source_soup.find("a")["name"]
        source_str_dict[cand_id] = source_str

    # Build the ordered file
    file_str = start_file_str

    all_priority = []

    for group_name, df_group in metadata.groupby("rank"):

        file_str += heading_format_str.format(group_name)
        for i, row in df_group.iterrows():
            cand_id = str(row["id"])
            if cand_id not in source_str_dict.keys():
                continue
            source_lines = source_str_dict[cand_id]
            add_robot_scores(source_lines, row["robot_str_g"], row["robot_str_i"])
            file_str += "".join(source_lines)
            all_priority.append(row["id"])
        file_str += "</br></br></br></br></br>"

    file_str += heading_format_str.format("Other Sources")
    for cand_id in source_str_dict:
        if cand_id in all_priority:
            continue
        else:
            file_str += "".join(source_str_dict[cand_id])
    file_str += end_file_str

    return file_str


def add_robot_scores(source_lines, g_str, i_str):
    for i, line in enumerate(source_lines):
        if "CCD#" in line:
            ccd_row_index = i
            break

    robot_str = """<tr>
    <td ALIGN="left" VALIGN="top" NOSAVE>Robot (g)</td>
    <td ALIGN="left" VALIGN="top" NOSAVE>{}</td>
    <tr>
    <td ALIGN="left" VALIGN="top" NOSAVE>Robot (i)</td>
    <td ALIGN="left" VALIGN="top" NOSAVE>{}</td>
    """.format(
        g_str, i_str
    )

    source_lines.insert(ccd_row_index + 3, robot_str)


def replace_links(file_str, junk_path, new_path="/"):
    replaced = re.sub(junk_path, new_path, file_str)
    replaced = re.sub("<br>http[^<]*", "", replaced)
    return replaced


def process_file(
    in_path,
    out_path,
    metadata,
    junk_path="https://stsci-transients.stsci.edu/[^/]*/",
    new_path="http://kntrap-bucket.s3-website-us-east-1.amazonaws.com/",
):
    print(f"Processing file {in_path}")
    file_str = build_ordered_file(in_path, metadata)
    file_str = replace_links(file_str, junk_path, new_path=new_path)
    f = open(out_path, "w")
    f.write(file_str)
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reorder html")

    parser.add_argument(
        "--path_metadata",
        type=str,
        default="./Fink_outputs",
        help="Path to Fink outputs",
    )
    parser.add_argument(
        "--path_data",
        type=str,
        default="./data/caldat20220214/",
        help="Path to Fink outputs",
    )
    parser.add_argument(
        "--field", type=str, default="353A_tmpl", help="Field name",
    )

    args = parser.parse_args()

    # read metadata
    fname_metadata = glob.glob(f"{args.path_metadata}/selected_{args.field}*.csv")[-1]
    metadata = pd.read_csv(fname_metadata, sep=";")
    metadata["rank"] = metadata.index.copy()

    ccd = 16
    runnumber = 8
    # change order in YSE piepline html
    fname_html = f"{args.path_data}/{args.field}/{ccd}/{args.field.strip('_tmpl')}_{ccd}.{runnumber}/index.html"
    out_fname_html = f"{args.path_data}/{args.field}/index_new.html"
    process_file(fname_html, out_fname_html, metadata)

    print(f"Finished new html {out_fname_html}")

