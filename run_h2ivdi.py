"""Program that runs HiVDI on sets of reaches and writes a NetCDF file that
contains A0, n, and Q time series.
"""

# Standard imports
import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import traceback
import warnings
import glob

# Third party imports
from netCDF4 import Dataset
from numpy import linspace,reshape,diff,ones,array,empty,mean,zeros,putmask
import numpy as np
import numpy.ma as ma

# Application imports
from HiVDI.core.utils.logging_utils import RichLogger, ServerLogger
from HiVDI.processors.confluence_case_processors import ConfluenceReachCaseProcessor

FILLVALUE = -999999999999

def create_empty_estimates_dict(reach_id):
    """Create an empty estimates dictionnary.

    Parameters
    ----------
    reach_id: int
        Unique reach identifier
    
    Returns
    -------
    list
        List of estimates dictionaries with empty data
    """
    return {"reach_id" : reach_id,
            "x" : np.array([np.nan]),
            "alpha": np.array([np.nan]),
            "beta": np.array([np.nan]),
            "A0" : np.array([np.nan]),
            "t" : np.array([np.nan]),
            "Q": np.array([np.nan]),
            "flags": (0,0),
            "exception" : 0,
            "status" : 0}

# DEPRECATED
def create_empty_estimates(reach_id):
    """Create an empty estimates list for invalid reach data.

    Parameters
    ----------
    reach_id: int
        Unique reach identifier
    
    Returns
    -------
    list
        List of estimates dictionaries with empty data
    """
    return [{
        "reach_id" : reach_id,
        "x" : ma.masked_array(np.array([-1.0]), mask=[1], fill_value=FILLVALUE),
        "alpha": ma.masked_array(np.array([-1.0]), mask=[1], fill_value=FILLVALUE),
        "beta": ma.masked_array(np.array([-1.0]), mask=[1], fill_value=FILLVALUE),
        "A0" : ma.masked_array(np.array([-1.0]), mask=[1], fill_value=FILLVALUE),
        "t" : np.array([np.nan]),
        "Q": np.array([np.nan]),
        "flags": (0,0)
    }]

def get_reachids(reachjson):
    """Extract and return a list of reach identifiers from json file.
    
    Parameters
    ----------
    reachjson : str
        Path to the file that contains the list of reaches to process
    
        
    Returns
    -------
    list
        List of reaches identifiers
    """

    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    with open(reachjson) as jsonfile:
        data = json.load(jsonfile)
    return data[index]["reach_id"]


def get_reach_dataset(reachjson):
    """Extract and return dataset associated to a reach from json file and AWS_BATCH_JOB_ARRAY_INDEX
    
    Parameters
    ----------
    reachjson : str
        Path to the file that contains the list of reaches to process
    
        
    Returns
    -------
    dict
        Reach dataset
    """

    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    with open(reachjson) as jsonfile:
        data = json.load(jsonfile)
    return data[index]


def reaches_batch_process(reach_datasets, inputdir, rundir, chain="classic", clean=True):
    """Process a batch of reaches
    
    
    Parameters
    ----------
    reachids : list
        List of reaches datasets
    inputdir : str
        Path to the input directory
    rundir : str
        Path to the run directory
    chain : str
        Chain to use, possible choices are: "classic" or "surrogate"
    clean : bool
        True to clean cases directories in the run directory
        
    Returns
    -------
    list
        List of estimates dictionaries
    """
    
    logger = logging.getLogger("HiVDI")

    # Loop on reaches
    estimates = []
    for reach_dataset in reach_datasets:
        
        # Retrieve dataset definitions
        reachid = reach_dataset["reach_id"]
        swot_file = reach_dataset["swot"]
        sos_file = reach_dataset["sos"]
        sword_file = reach_dataset["sword"]

        logger.section_start("Process reach %s" % reachid)
        
        # Create empty estimates dictionnary
        estimate = create_empty_estimates_dict(reachid)

        # Create parameters dictionary
        parameters_dict = {"case_name" : "%s" % reachid,
                           "chain" : chain,
                           "reach_id" : int(reachid),
                           "upstream_area" : 1000000,           # TODO get real values from SOS or other DB
                           "dx" : 50,
                           "sword_file" : os.path.join(inputdir, "sword", sword_file),
                           "swot_file" : os.path.join(inputdir, "swot", swot_file),
                           "sos_file" : os.path.join(inputdir, "sos", sos_file),
                           "output_file" : None,
                           "data_level" : "node",
                           "bc_out" : "heights",
                           "run_dir" : rundir}
        
        # Create processor
        valid_run = True
        try:
            case_processor = ConfluenceReachCaseProcessor(parameters_dict, debug=False)
        except Exception as err:
            logger.error(err)
            logger.error(traceback.format_exc())
            estimate["exception"] : 1
            valid_run = False
            
        # Handle case with non loaded observations
        if valid_run is True and case_processor.obs_validity == -9:
            valid_run = False
        
        # Update estimates dict with time occurences
        if valid_run is True:
            estimate["t"] = case_processor.obs.t / 86400.0
            estimate["Q"] = ones(case_processor.obs.t.size) * np.nan
        
        # Handle case with non valid observations
        if valid_run is True and case_processor.obs_validity == -1:
            valid_run = False

        # Run processor
        if valid_run is True:
            try:
                case_processor.run(reset=1)
            except Exception as err:
                logger.error(err)
                logger.error(traceback.format_exc())
                estimate["exception"] : 1
                valid_run = False
        
        # Post-process results to get estimates
        if valid_run is True:
            try:
                x, alpha, beta, A0, t, Q, flags = case_processor.post(write_file=False)
                estimate["x"] = x
                estimate["alpha"] = alpha
                estimate["beta"] =  beta
                estimate["A0"] =  A0
                estimate["Q"] =  Q
                estimate["flags"] = flags
                estimate["status"] = 1
            except Exception as err:
                logger.error(err)
                estimate["exception"] : 1
                valid_run = False
        
        # Append estimates dictionary in list
        estimates.append(estimate)

        # Clean run_dir
        if clean:
            if os.path.isdir(os.path.join(rundir, "%s" % reachid)):
                shutil.rmtree(os.path.join(rundir, "%s" % reachid))

        if valid_run is True:
            logger.section_end("Process reach %s ( SUCCESS )" % reachid)
        else:
            logger.section_end("Process reach %s ( FAILED )" % reachid)
        
    return estimates


def write_output(estimates, outputdir):
    """Write HiVDI estimates to NetCDF files in output directory.
    
    Parameters
    ----------
    estimates: list
        List of estimates dictionaries
    outputdir : str
        Path to the output directory
    """
    
    logger = logging.getLogger("HiVDI")
    
    for estimate in (estimates):
        
        # Retrieve reachid, days and flags
        reachid = estimate["reach_id"]
        t = estimate["t"]
        flags = estimate["flags"]
        exception = estimate["exception"]
        status = estimate["status"]

        # Create output dataset
        logger.info("Output estimates for reach %s" % reachid)
        outfile = os.path.join(outputdir, "%s_hivdi.nc" %reachid)
        dataset = Dataset(outfile, 'w', format="NETCDF4")
        
        # Set global attributes
        dataset.setncatts({"title" : "HIVDI output for reach ID: %s" % reachid,
                           "production_date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "reach_id" : "%s" % reachid,
                           "exception" : exception,
                           "obs_validity" : flags[0],
                           "VDA_status" : flags[1],
                           "status" : status})

        # Create dimensions
        dataset.createDimension("nchar", 12)
        dataset.createDimension("nt", len(t))
        
        # Create global variables
        nt = dataset.createVariable("nt", "i4", ("nt",))
        nt.units = "day"
        nt.long_name = "nt"
        nt[:] = t

        # Create group
        reach_group = dataset.createGroup("reach")
        
        # Create variables in reach group
        reach_id = reach_group.createVariable("reach_id", "c", ("nchar",))
        reach_id_str = "%i" % reachid
        for i in range(0, min(len(reach_id_str), 12)):
            reach_id[i] = reach_id_str[i]
        A0 = reach_group.createVariable("A0", "f8", (), fill_value=FILLVALUE)
        A0.long_name = "unobserved cross-sectional area"
        A0.units = "m^2"
        A0[0] = np.nan_to_num(estimate["A0"][0], copy=True, nan=FILLVALUE)
        alpha = reach_group.createVariable("alpha", "f8", (), fill_value=FILLVALUE)
        alpha.long_name = "coefficient for the Strickler power law"
        alpha.units = "m^(1/3)/s" 
        alpha[0] = np.nan_to_num(estimate["alpha"][0], copy=True, nan=FILLVALUE)
        beta = reach_group.createVariable("beta", "f8", (), fill_value=FILLVALUE)
        beta[0] = np.nan_to_num(estimate["beta"][0], copy=True, nan=FILLVALUE)
        beta.long_name = "exponent for the Strickler power law"
        beta.units = "-" 
        Q = reach_group.createVariable("Q", "f8", ("nt",), fill_value=FILLVALUE)
        Q.long_name = "discharge"
        Q.units = "m^3/s" 
        Q[:] = np.nan_to_num(estimate["Q"][:], copy=True, nan=FILLVALUE)
        
        # Close output dataset
        dataset.close()

if __name__ == "__main__":

    inputdir = "/mnt/data/input"
    outputdir = "/mnt/data/output"
    rundir = "/app/HiVDI/run_dir"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run HiVDI for Confluence")
    parser.add_argument("-reachjson", type=str, default="reaches.json",
                        help="File containing reaches list")
    parser.add_argument("-chunk", type=int,
                        help="Chunk to process")
    parser.add_argument("-chain", type=str, default="classic",
                        choices=["classic", "surrogate"],
                        help="Chain to use")
    args = parser.parse_args()
    all_reach_jsons = glob.glob(os.path.join(inputdir, 'reaches*'))
    reachjson = all_reach_jsons[args.chunk]
    print('Processing', reachjson)
    reach_dataset = get_reach_dataset(reachjson)
    
    # Load log config
    if "HIVDI_LOG_CONFIG" in os.environ:
        log_config = os.environ["HIVDI_LOG_CONFIG"]
        if log_config == "rich-term":
            logging.setLoggerClass(RichLogger)
        elif log_config == "server":
            logging.setLoggerClass(ServerLogger)
        else:
            raise ValueError("Wrong value of environment variable HIVDI_LOG_CONFIG : %s" % log_config)
    else:
        logging.setLoggerClass(RichLogger)
    
    # Setup logger
    logger = logging.getLogger("HiVDI")
    if "HIVDI_LOG_FILE" in os.environ:
        log_file = os.environ["HIVDI_LOG_FILE"]
    else:
        log_file = None
    if "HIVDI_LOG_LEVEL" in os.environ:
        log_level = os.environ["HIVDI_LOG_LEVEL"]
    else:
        log_level = "info"
    logger.setup(log_level, log_file)

    # Setup logging for PyMC3
    pymc3_logger = logging.getLogger("pymc3")
    pymc3_logger.setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Process batch of reaches
    logger.section_start("Process CONFLUENCE batch")
    try:
        estimates = reaches_batch_process([reach_dataset], inputdir, rundir, chain=args.chain, clean=True)
    except Exception as err:
        track = traceback.format_exc()
        logger.error(track)
        raise
    logger.section_end("Process CONFLUENCE batch")
    
    # Output estimates
    logger.section_start("Output CONFLUENCE batch estimates")
    try:
        write_output(estimates, outputdir)
    except Exception as err:
        track = traceback.format_exc()
        logger.error(track)
        raise
    logger.section_end("Output CONFLUENCE batch estimates")

