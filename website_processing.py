import re
import glob
import shutil
import argparse
import pandas as pd
from bs4 import BeautifulSoup
import os

""" Author Dougal Dobie

    Use metadata (ranked candidates) to reorder YSE pipeline webportal html
"""


def build_ordered_file(filename, metadata):
    heading_format_str = (
        '<center><b><font color="black"><font size=+4>{}</font></font></b>\n'
    )
    field_info = filename.split('/')[-2]
    field_name = field_info.split('_')[0]
    ccd_num = field_info.split('_')[1].split('.')[0]
    run_num = field_info.split('_')[1].split('.')[1]
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
    
    incl_robot = False
    if "ROBOT_scores_g" in metadata.columns:
        incl_robot=True
    for group_name, df_group in metadata.groupby("rank"):

        file_str += heading_format_str.format(group_name)
        for i, row in df_group.iterrows():
            cand_name = str(row["id"])
            cand_id = cand_name.split('_')[-1].replace('cand','')
            fink_field_info = "_".join(cand_name.split('_')[0:2])
            #print(fink_field_info, field_info)
            
            if cand_id not in source_str_dict.keys():
                continue
            if field_info != fink_field_info:
                continue
            source_lines = source_str_dict[cand_id]
            #if incl_robot:
            #    add_robot_scores(source_lines, row["ROBOT_scores_g"], row["ROBOT_scores_i"])
            if incl_robot:
                temp = row["ROBOT_scores_g"]
                if len(temp.replace('[','').replace(']',''))  > 0:
                    robot_g_scores = [f'{float(x):0.3f}' for x in temp.replace('[','').replace(']','').split(',')]
                else:
                    robot_g_scores = temp
                temp = row["ROBOT_scores_i"]
                if len(temp.replace('[','').replace(']',''))  > 0:
                    robot_i_scores = [f'{float(x):0.3f}' for x in temp.replace('[','').replace(']','').split(',')]
                else:
                    robot_i_scores = temp
                add_robot_scores(source_lines, f"{robot_g_scores}", f"{robot_i_scores}")
            source_text = "".join(source_lines)
            
            # Replace colours
            source_text = source_text.replace("background-color:lightblue","background-color:#228B22")
            
            file_str += source_text
            all_priority.append(cand_id)
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
    metadata,
    junk_path="https://stsci-transients.stsci.edu/[^/]*/",
    new_path="http://kntrap-bucket.s3-website-us-east-1.amazonaws.com/",
):
    print(f"Processing file {in_path}")
    fname_html = in_path

    # keep a copy with old ordered data
    fname_html_old = fname_html.replace(".html", "_old.html")
    if not os.path.isfile(fname_html_old):
        shutil.copy(fname_html, fname_html_old)

    file_str = build_ordered_file(fname_html_old, metadata)
    file_str = replace_links(file_str, junk_path, new_path=new_path)

    # save new file with same name
    f = open(fname_html, "w")
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
    metadata["rank"] = "Priority"#metadata.index.copy()

    index_path_list = glob.glob(f"{args.path_data}/{args.field}/*/*/index.html")
    for fname_html in index_path_list:
        
        process_file(fname_html, metadata)

    print(f"Finished new html")

