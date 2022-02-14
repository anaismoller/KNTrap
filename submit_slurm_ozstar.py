#!/usr/bin/env python

""" submit_slurm_ozstar.py -- Input fieldname, run, create a bash shell script to run the entire KNTraP pipeline, and submit it as a slurm job on OzStar. 

Usage: submit_slurm_ozstar [-h] [-q] [-v] [--debug] [--overwrite] [--do_not_submit] [--kntrap_sel_path STRING] [--conda_env_name STRING] [--kntrap_data_dir STRING] [--outdir STRING] [--ozstar_reservation STRING] <fieldname> <run>

Arguments:
    fieldname (string)
    run (string)

Options:
    -h, --help                          Show this screen
    -q, --quietmode                     Minimize print to screen. This is useful when this function is called in another function. [default: False]  
    -v, --verbose                       Print extra info to screen. [default: False]
    --debug                             Print debugging info to screen. [default: False]
    --overwrite                         Overwrite any existing files at destination [default: False]
    --do_not_submit                     Just write the slurm script and pipeline bash scripts, don't submit via sbatch [default: False]
    --kntrap_sel_path STRING            Where src for KNTraP project lives [default: /fred/oz100/NOAO_archive/KNTraP_Project/src/KNTraP/]
    --conda_env_name STRING             Python conda environment name [default: anais]
    --kntrap_data_dir STRING            KNTraP data and working directory [default: /fred/oz100/NOAO_archive/KNTraP_Project/kntrappipe]
    --outdir STRING                     Output the bash script here. If not set, will output in kntrap_data_dir/logs/ozstar/<fieldname>_<run>
    --ozstar_reservation STRING         If set, in sbatch script put #SBATCH --reservation={ozstar_reservation}

Examples:
    python submit_slurm_ozstar.py GRB210605A5
"""
import docopt
import sys, os

# KNTraP modules
from utils import write_kntrap_bashscript

__author__ = "A Moller modified from Jielai Zhang"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "2021-02-15"

##############################################################
####################### Main Function ########################
##############################################################

batch_script_template = """#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --output=/KNTRAP_SELECTION_DIR/logs/kntrappipe_JOB_NAME.out
#SBATCH --error=/KNTRAP_SELECTION_DIR/logs/kntrappipe_JOB_NAME.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00
#SBATCH --mem-per-cpu=800
RESERVATION_LINE

echo Slurm Job JOB_NAME start
echo Job bash script is: JOB_BASH_SCRIPT
echo Job .out is saved at: /KNTRAP_SELECTION_DIR/logs/JOB_NAME.out
echo Job .err is saved at: /KNTRAP_SELECTION_DIR/logs/JOB_NAME.err
echo `date`
SECONDS=0
echo -------- --------
bash JOB_BASH_SCRIPT
echo -------- --------
echo `date`
duration=$SECONDS
echo Slurm Job JOB_NAME done in $(($duration / 60)) minutes and $(($duration % 60)) seconds
"""


def submit_slurm_ozstar(
    fieldname,
    run,
    kntrap_sel_path="/fred/oz100/NOAO_archive/KNTraP_Project/photpipe/v20.0/DECAMNOAO/KNTraP/KNTrap_selection",
    conda_env_name="kntrap",
    kntrap_data_dir="/fred/oz100/NOAO_archive/KNTraP_Project/photpipe/v20.0/DECAMNOAO/KNTraP/web/web/sniff/",
    outdir=None,
    ozstar_reservation=None,
    verbose=False,
    debugmode=False,
    quietmode=False,
    do_not_submit=False,
    overwrite=False,
):

    # Create the bash script
    kntrap_bashscript_path = write_kntrap_bashscript.write_kntrap_bashscript(
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
    )  # overwrite function not implemented yet

    # Define slurm job name
    slurm_job_name = f"{fieldname}_{run}"

    # Figure out where to save the bash script
    slurm_script_dir = kntrap_sel_path + "/logs/"
    slurm_script_path = slurm_script_dir + f"/kntrappipe_{slurm_job_name}_slurm.sh"

    # Create output directory if not exist
    os.makedirs(slurm_script_dir, exist_ok=True)

    # Create slurm batch bash script
    script_string = batch_script_template.replace("JOB_NAME", slurm_job_name)
    script_string = script_string.replace("PIPE_DATA_DIR", kntrap_data_dir)
    script_string = script_string.replace("JOB_BASH_SCRIPT", kntrap_bashscript_path)
    script_string = script_string.replace("FIELDNAME", fieldname)
    script_string = script_string.replace("RUN", run)
    if ozstar_reservation == None:
        script_string = script_string.replace("RESERVATION_LINE", "")
    else:
        script_string = script_string.replace(
            "RESERVATION_LINE", f"#SBATCH --reservation={ozstar_reservation}"
        )

    # Write the bash script to file
    f = open(slurm_script_path, "w")
    f.write(script_string)
    f.close()

    # print if not quietmode
    if not quietmode:
        print(f"Saved: {slurm_script_path}")

    # submit slurm script
    if do_not_submit == False:
        command = f"sbatch {slurm_script_path}"
        print(f"Running: {command}")
        try:
            os.system(command)
            if verbose:
                print("VERBOSE: Useful slurm queue check commands:")
                print(
                    'VERBOSE: alias watchnodes: squeue --user=fstars -i5 --format="%.11i %.9P %.29j %.8u %.7T %.8M %.4D %R"'
                )
                print(
                    'VERBOSE: alias printnodes: squeue --user=fstars --format="%.11i %.9P %.29j %.8u %.7T %.8M %.4D %R"'
                )
        except:
            sys.exit(f"!!! ERROR-- sys.exit when running: {command}")

    # Finish
    return slurm_script_path


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
    do_not_submit = arguments["--do_not_submit"]
    # Required arguments
    fieldname = arguments["<fieldname>"]
    run = arguments["<run>"]
    # Optional arguments (with defaults set)
    kntrap_sel_path = arguments["--kntrap_sel_path"]
    conda_env_name = arguments["--conda_env_name"]
    kntrap_data_dir = arguments["--kntrap_data_dir"]
    outdir = arguments["--outdir"]
    ozstar_reservation = arguments["--ozstar_reservation"]
    # Not implemented arguments (to be implemented later)
    overwrite = arguments["--overwrite"]

    _ = submit_slurm_ozstar(
        fieldname,
        run,
        kntrap_sel_path=kntrap_sel_path,
        conda_env_name=conda_env_name,
        kntrap_data_dir=kntrap_data_dir,
        outdir=outdir,
        ozstar_reservation=ozstar_reservation,
        verbose=verbose,
        debugmode=debugmode,
        quietmode=quietmode,
        do_not_submit=do_not_submit,
        overwrite=overwrite,
    )
    # overwrite function not yet fully implemented.
