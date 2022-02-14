#!/usr/bin/env python

""" write_kntrap_bashscript.py -- Input fieldname, filterband, ctio_caldate, fitsextension, create a bash shell script to run the entire KNTraP pipeline. 

Usage: write_kntrap_bashscript [-h] [-q] [-v] [--debug] [--overwrite] [--kntrap_sel_path STRING] [--conda_env_name STRING] [--kntrap_data_dir STRING] [--outdir STRING] <fieldname> <ctio_caldate> <filterband> <fitsextension> 

Arguments:
    fieldname (string)
    ctio_caldate (string)
        e.g. 20210607
    filterband (string)
        e.g. g or r or i
    fitsextension (string)
        e.g. 30, can be 1-61

Options:
    -h, --help                          Show this screen
    -q, --quietmode                     Minimize print to screen. This is useful when this function is called in another function. [default: False]  
    -v, --verbose                       Print extra info to screen. [default: False]
    --debug                             Print debugging info to screen. [default: False]
    --overwrite                         Overwrite any existing files at destination [default: False]
    --kntrap_sel_path STRING            Where src for KNTraP project lives [default: ./src/KNTraP/]
    --conda_env_name STRING             Python conda environment name [default: kntrap]
    --kntrap_data_dir STRING            KNTraP data and working directory [default: ./kntrappipe]
    --outdir STRING                     Output the bash script here. If not set, will output in kntrap_data_dir/logs/ozstar/<fieldname>/. 

Examples:
    python write_kntrap_bashscript.py GRB210605A5 g 20210607 30
"""
import os
import docopt

__author__ = "Jielai Zhang"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "2021-01-20"
__maintainer__ = "Jielai Zhang"
__email__ = "zhang.jielai@gmail.com"

##############################################################
####################### Main Function ########################
##############################################################

script_template_no_spreadmodel = """# No Spread Model Version

cd PIPE_DATA_DIR
source activate CONDA_ENV_NAME
export PYTHONPATH=$PYTHONPATH:SRC_DIR

src_dir=SRC_DIR
field=FIELDNAME
caldate=CTIO_CALDATE # calendar date at start of night at telescope
band=FILTERBAND
ext=FITSEXTENSION

#STEP1: split CPTrigger stack fits file's 61 exts into 61 fits files
# This was done manually as it involved for NOAO archive data getting instcal, then splitting into CCDs then stacking after splitting it. 
# This would not be the observing run procedure.

# ===========================================================================
# For each DECam pointing, every step below this note
# needs to be done for each CCD split out in #STEP1.
# ===========================================================================

echo kntrappipe bash script starting Using:
echo src_dir       = ${src_dir}
echo field         = ${field}
echo ctio_caldate  = ${caldate}
echo band          = ${band}
echo ext           = ${ext}
echo conda_env     = CONDA_ENV_NAME
echo pipe_data_dir = PIPE_DATA_DIR


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "ALIGN template and science"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
#STEP2: align template and science image, output template_resamp.fits and science_resamp.fits
# overall syntax: align_image --swarp swarp_loc -o save_in_this_dir template.fits science.fits
echo "Execute: python ${src_dir}/align_image.py --swarp swarp -o data_outputs/${field}/${caldate} TEMPLATE_IMAGES/${field}_${band}_stacked_template.fits data_unpacked/${field}/${field}_${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.fits"
python ${src_dir}/align_image.py --swarp swarp -o data_outputs/${field}/${caldate} TEMPLATE_IMAGES/${field}_${band}_stacked_template.fits data_unpacked/${field}/${field}_${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.fits
echo "Execute: mv data_outputs/${field}/${caldate}/${field}_${band}_stacked_template.resamp.fits data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits"
mv data_outputs/${field}/${caldate}/${field}_${band}_stacked_template.resamp.fits data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits
echo "||| ALIGN template and science Target file created? --> ||||"
ls data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits
ls data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp.fits


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "SUBTRACT: science - template"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP3: subtract template and science image, output subtraction.fits
# overall syntax: subtract_image --sextractor SE_loc template_resamp.fits and science_resamp.fits -s subtraction.fits 
echo "Execute: python ${src_dir}/subtract_image.py -o --sextractor sex data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp.fits -s data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits"
python ${src_dir}/subtract_image.py -o --sextractor sex data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp.fits -s data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits
echo "||| ALIGN template and science Target file created? --> ||||"
ls data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "SOURCE EXTRACT: template"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP4: source extract template image, output template.cat
# overall syntax: run_sourceextractor -s SE_loc -p PSFEX_loc --options --savecats OUTPUT_DIR template.fits
echo "Execute: python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending TEMPLATE --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits"
python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending TEMPLATE --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${band}_stacked_template_ext${ext}.resamp.fits


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "SOURCE EXTRACT: science"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP5: source extract science image, output science.cat
# overall syntax: run_sourceextractor -s SE_loc -p PSFEX_loc --options --savecats OUTPUT_DIR science.fits
echo "Execute: python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending SCI --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate}  data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp.fits"
python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending SCI --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate}  data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp.fits


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "SOURCE EXTRACT: subtraction image"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP6: source extract subtraction image, output subtraction.cat
# overall syntax: run_sourceextractor -s SE_loc -p PSFEX_loc --options --savecats OUTPUT_DIR subtraction.fits
# !!!! This step takes in the output from pre-step6 instead of what it used to take in.
echo "Execute: python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending SUB --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits"
python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending SUB --fwhm 1.1 --detect_minarea 10 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits 


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "INVERT subtraction image"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP7: invert the subtraction image , output inv_subtraction.fits
# overall syntax: invert -o OUTPUT_DIR --options subtraction.fits
echo "Execute: python ${src_dir}/invert_fits.py -o data_outputs/${field}/${caldate} --overwrite data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits"
python ${src_dir}/invert_fits.py -o data_outputs/${field}/${caldate} --overwrite data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub.fits


echo ''
echo "||||||||||||||||||||||||||||||||||||||||||||||"
echo "SOURCE EXTRACT: inverted subtraction or sub_neg image"
echo "||||||||||||||||||||||||||||||||||||||||||||||"
# STEP8: source extract inverted subtraction image, output inv_subtraction.cat
# overall syntax: run_sourceextractor -s SE_loc -p PSFEX_loc --options --savecats OUTPUT_DIR subtraction.fits 
echo "Execute: python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending NEG --fwhm 1.1 --detect_minarea 8 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub_neg.fits"
python ${src_dir}/run_sourceextractor.py -v -s sex -p psfex --catending NEG --fwhm 1.1 --detect_minarea 8 --detect_thresh 1.0 --savecats data_outputs/${field}/${caldate} data_outputs/${field}/${caldate}/${field}_${caldate}_${band}_stack_ext${ext}.resamp_sub_neg.fits


# below not edited yet

# STEP9: read subtraction.cat, inverted sub, sci, templ catalogues and deteremine best new transient candidates; outputs 3 ds9 region files
# overall syntax: find_new_transients -o OUTPUT_DIR sub.cat sci.cat inverted_sub.cat template.cat
# python /fred/oz100/jielaizhang/src/dataexplore/datavis/ascii/find_new_transientCandidates_DECam.py -v -o 20210605/GRB210605Agreen1/g/ 20210605/GRB210605Agreen1/g/GRB210605Agreen1_20210605_g_stack_ext5.resamp_sub_withnans_SUB.cat 20210605/GRB210605Agreen1/g/GRB210605Agreen1_20210605_g_stack_ext5.resamp_withnans_SCI.cat 20210605/GRB210605Agreen1/g/GRB210605Agreen1_20210605_g_stack_ext5.resamp_sub_withnans_neg_NEG.cat 20210605/GRB210605Agreen1/g/GRB210605Agreen1_g_stacked_temp.resamp_TEMPL.cat

"""


def write_kntrap_bashscript(
    fieldname,
    kntrap_sel_path="./src/KNTraP/",
    conda_env_name="kntrap",
    kntrap_data_dir="./kntrappipe",
    outdir=None,
    verbose=False,
    debugmode=False,
    quietmode=False,
    overwrite=False,
):

    # Create the bash script
    script_string = script_template_no_spreadmodel.replace(
        "PIPE_DATA_DIR", kntrap_data_dir
    )
    script_string = script_string.replace("CONDA_ENV_NAME", conda_env_name)
    script_string = script_string.replace("SRC_DIR", kntrap_sel_path)
    script_string = script_string.replace("FIELDNAME", fieldname)

    # Figure out where to save the bash script
    if outdir == None:
        bash_script_dir = kntrap_data_dir + f"/logs/ozstar/{fieldname}/"
    else:
        bash_script_dir = outdir
    bash_script_path = bash_script_dir + f"/kntrappipe_{fieldname}.sh"

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
    ctio_caldate = arguments["<ctio_caldate>"]
    filterband = arguments["<filterband>"]
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
        ctio_caldate,
        filterband,
        fitsextension,
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
