#!/usr/bin/env python

""" submit_list_fiels_ozstar.py -- Input runnumber, fieldnames create a bash shell script to run the entire KNTraP pipeline, and submit it as a slurm job on OzStar. 

Usage: submit_slurm_ozstar [-h] [-q] [-v] [--debug] [--overwrite] [--do_not_submit] [--kntrap_sel_path STRING] [--conda_env_name STRING] [--kntrap_data_dir STRING] [--outdir STRING] [--pathout STRING] [--ozstar_reservation STRING] <fieldnames>  <runnumber>

Arguments:
    fieldnames (string)
    runnumber (string)

Options:
    -h, --help                          Show this screen
    -q, --quietmode                     Minimize print to screen. This is useful when this function is called in another function. [default: False]  
    -v, --verbose                       Print extra info to screen. [default: False]
    --debug                             Print debugging info to screen. [default: False]
    --overwrite                         Overwrite any existing files at destination [default: False]
    --do_not_submit                     Just write the slurm script and pipeline bash scripts, don't submit via sbatch [default: False]
    --kntrap_sel_path STRING            Where src for KNTraP project lives [default: /fred/oz100/NOAO_archive/KNTraP_Project/photpipe/v20.0/DECAMNOAO/KNTraP/KNTrap_selection]
    --conda_env_name STRING             Python conda environment name [default: anais]
    --kntrap_data_dir STRING            KNTraP data and working directory [default: /fred/oz100/NOAO_archive/KNTraP_Project/photpipe/v20.0/DECAMNOAO/KNTraP/web/web/sniff/]
    --outdir STRING                     Output the bash script here. If not set, will output in kntrap_data_dir/logs/ozstar/<fieldname>_<run>
    --pathout STRING                   Output the processed data [default: /fred/oz100/NOAO_archive/KNTraP_Project/photpipe/v20.0/DECAMNOAO/KNTraP/KNTrap_selection/Fink_outputs]
    --ozstar_reservation STRING         If set, in sbatch script put #SBATCH --reservation={ozstar_reservation}

Examples:
    python submit_slurm_ozstar.py 12 'GRB210605A5 fielda fieldb'
"""
import docopt
from submit_slurm_ozstar import submit_slurm_ozstar

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
    fieldnames = arguments["<fieldnames>"]
    runnumber = arguments["<runnumber>"]
    # Optional arguments (with defaults set)
    kntrap_sel_path = arguments["--kntrap_sel_path"]
    conda_env_name = arguments["--conda_env_name"]
    kntrap_data_dir = arguments["--kntrap_data_dir"]
    outdir = arguments["--outdir"]
    pathout = arguments["--pathout"]
    ozstar_reservation = arguments["--ozstar_reservation"]
    # Not implemented arguments (to be implemented later)
    overwrite = arguments["--overwrite"]

    for fieldname in fieldnames.split(" "):
        print("Submitting field", fieldname)
        _ = submit_slurm_ozstar(
            fieldname,
            runnumber,
            kntrap_sel_path=kntrap_sel_path,
            conda_env_name=conda_env_name,
            kntrap_data_dir=kntrap_data_dir,
            outdir=outdir,
            pathout=pathout,
            ozstar_reservation=ozstar_reservation,
            verbose=verbose,
            debugmode=debugmode,
            quietmode=quietmode,
            do_not_submit=do_not_submit,
            overwrite=overwrite,
        )
