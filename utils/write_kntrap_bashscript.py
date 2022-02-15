#!/usr/bin/env python

""" write_kntrap_bashscript.py -- Input fieldname, create a bash shell script to run the entire KNTraP pipeline. 

Usage: write_kntrap_bashscript [-h] [-q] [-v] [--debug] [--overwrite] [--kntrap_sel_path STRING] [--conda_env_name STRING] [--kntrap_data_dir STRING] [--outdir STRING] <fieldname> <run> 

Arguments:
    fieldname (string)
    run (string)

Options:
    -h, --help                          Show this screen
    -q, --quietmode                     Minimize print to screen. This is useful when this function is called in another function. [default: False]  
    -v, --verbose                       Print extra info to screen. [default: False]
    --debug                             Print debugging info to screen. [default: False]
    --overwrite                         Overwrite any existing files at destination [default: False]
    --kntrap_sel_path STRING            Where src for KNTraP project lives [default: ./src/KNTraP/]
    --conda_env_name STRING             Python conda environment name [default: anais]
    --kntrap_data_dir STRING            KNTraP data and working directory [default: ./kntrappipe]
    --outdir STRING                     Output the bash script here. If not set, will output in kntrap_sel_path/logs/ozstar/<fieldname>

Examples:
    python write_kntrap_bashscript.py GRB210605A5
"""
import os
import docopt

__author__ = "A Moller modified from Jielai Zhang"
__license__ = "MIT"
__version__ = "1.0.1"

##############################################################
####################### Main Function ########################
##############################################################

script_template_no_spreadmodel = """# No Spread Model Version

cd PIPE_DATA_DIR
source activate CONDA_ENV_NAME
export PYTHONPATH=$PYTHONPATH:SRC_DIR

python src_dir/run.py --path_field kntrap_data_dir/fieldname --run runnumber --path_out pathoutname

"""


def write_kntrap_bashscript(
    fieldname,
    run,
    kntrap_sel_path="./KNTrap_selection",
    conda_env_name="anais",
    kntrap_data_dir="./data",
    outdir=None,
    verbose=False,
    debugmode=False,
    quietmode=False,
    overwrite=False,
):

    # Create the bash script
    script_string = script_template_no_spreadmodel.replace(
        "kntrap_data_dir", kntrap_data_dir
    )
    script_string = script_string.replace("CONDA_ENV_NAME", conda_env_name)
    script_string = script_string.replace("src_dir", kntrap_sel_path)
    script_string = script_string.replace("fieldname", fieldname)
    script_string = script_string.replace("runnumber", run)

    # Figure out where to save the bash script
    if outdir == None:
        bash_script_dir = kntrap_sel_path + f"/logs/ozstar/{fieldname}/"
    else:
        bash_script_dir = outdir
    bash_script_path = bash_script_dir + f"/kntrappipe_{fieldname}.sh"

    script_string = script_string.replace("pathoutname", bash_script_dir)

    # Create output directory if not exist
    os.makedirs(bash_script_dir, exist_ok=True)

    # Write the bash script to file
    f = open(bash_script_path, "w")
    f.write(script_string)
    f.close()

    # print if not quietmode
    if not quietmode:
        print(f"Saved: {bash_script_path}")

    # Finish
    return bash_script_path


############################################################################
####################### BODY OF PROGRAM STARTS HERE ########################
############################################################################

if __name__ == "__main__":

    # Read in input arguments
    arguments = docopt.docopt(__doc__)
    # Code running mode arguments
    debugmode = arguments["--debug"]
    if debugmode:
        print(arguments)
    verbose = arguments["--verbose"]
    quietmode = arguments["--quietmode"]
    # Required arguments
    fieldname = arguments["<fieldname>"]
    run = arguments["<run>"]
    fitsextension = arguments["<fitsextension>"]
    # Optional arguments (with defaults set)
    kntrap_sel_path = arguments["--kntrap_sel_path"]
    conda_env_name = arguments["--conda_env_name"]
    kntrap_data_dir = arguments["--kntrap_data_dir"]
    outdir = arguments["--outdir"]
    # Not implemented arguments (to be implemented later)
    overwrite = arguments["--overwrite"]

    _ = write_kntrap_bashscript(
        fieldname,
        run,
        kntrap_sel_path=kntrap_sel_path,
        conda_env_name=conda_env_name,
        kntrap_data_dir=kntrap_data_dir,
        outdir=outdir,
        verbose=verbose,
        debugmode=debugmode,
        quietmode=quietmode,
        overwrite=overwrite,
    )
    # overwrite function not yet fully implemented.
    # currently overwrite is on
