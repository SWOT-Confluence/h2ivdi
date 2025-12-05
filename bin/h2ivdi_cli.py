import logging
import argparse
import fcntl
import json
import numpy as np
import os
import pandas as pd
import traceback
import warnings
warnings.filterwarnings(action='ignore', module='scipy')
warnings.filterwarnings(action='ignore', module='statsmodel')
# warnings.filterwarnings("error")


from H2iVDI.core.logger import create_logger
from H2iVDI.core.errors import __critical_errors__
from H2iVDI.processors import *

def load_runs_file(runs_file: str):
    """ Load runs file

        Parameters
        ----------
        runs_file: str
            Path to the runs file

        Return
        ------
        runs_list: list
            List of runs definitions
    """

    if os.path.splitext(runs_file)[1] == ".json":
        with open(runs_file) as json_fp:
            runs_list = json.load(json_fp)
    return runs_list


def init_status_table_file(status_table_file: str):
    """ Initalise status table

        Parameters
        ----------
        status_table_file: str
            Path to the status table file
    """

    status_table_fp = open(status_table_file, "w")
    fcntl.flock(status_table_fp, fcntl.LOCK_EX)
    status_table_fp.write("RUN_INDEX;PREPRO;RUN;POSTPRO\n")
    fcntl.flock(status_table_fp, fcntl.LOCK_UN)
    status_table_fp.close()


def append_status_table_file(status_table_file: str, run_index: int, prepro_code=None, run_code=None, postpro_code=None):
    """ Append run status in status table

        Parameters
        ----------
        status_table_file: str
            Path to the status table file
        run_index: int
            Index of current run
        prepro_code: int
            Error code of the preprocessing step
        run_code: int
            Error code of the run step
        postpro_code: int
            Error code of the postprocessing step
    """

    status_table_fp = open(status_table_file, "a")
    fcntl.flock(status_table_fp, fcntl.LOCK_EX)
    status_table_fp.write("%i" % run_index)
    status_table_fp.write(";%i" % prepro_code)
    if run_code is None:
        status_table_fp.write(";;\n")
    else:
        status_table_fp.write(";%i" % run_code)
        if postpro_code is None:
            status_table_fp.write(";\n")
        else:
            status_table_fp.write(";%i\n" % postpro_code)
    fcntl.flock(status_table_fp, fcntl.LOCK_EX)
    status_table_fp.close()


def load_failed_runs_from_status_table_file(status_table_file: str):
    """ Load failed runs from status table

        Parameters
        ----------
        status_table_file: str
            Path to the status table file

        Return
        ------
        indices: list
            List of failed runs
    """

    status_table = pd.read_csv(status_table_file, sep=";")
    failed_table = status_table[~np.isfinite(status_table["POSTPRO"])]

    if "SET_INDEX" in failed_table.columns:

        # Old version dedicated to CONFLUENCE
        return list(failed_table["SET_INDEX"].values)
    
    else:

        return list(failed_table["RUN_INDEX"].values)


def update_status_table_file(status_table_file: str, run_index: int, prepro_code=None, run_code=None, postpro_code=None):
    """ Update run status in status table

        Parameters
        ----------
        status_table_file: str
            Path to the status table file
        run_index: int
            Index of run to update
        prepro_code: int
            Error code of the preprocessing step
        run_code: int
            Error code of the run step
        postpro_code: int
            Error code of the postprocessing step
    """

    status_table = pd.read_csv(status_table_file, sep=";")
    if "SET_INDEX" in status_table.columns:
        update_index = status_table[status_table["SET_INDEX"] == run_index].index[0]
    else: 
        update_index = status_table[status_table["RUN_INDEX"] == run_index].index[0]
    status_table.loc[update_index, "PREPRO"] = prepro_code
    status_table.loc[update_index, "RUN"] = run_code
    status_table.loc[update_index, "POSTPRO"] = postpro_code

    status_table.to_csv(status_table_file, sep=";", index=False)


def append_or_update_status_table_file(status_table_file: str, run_index: int, prepro_code=None, run_code=None, 
                                       postpro_code=None, update=False):
    """ Append run status in status table

        Parameters
        ----------
        status_table_file: str
            Path to the status table file
        run_index: int
            Index of current run
        prepro_code: int
            Error code of the preprocessing step
        run_code: int
            Error code of the run step
        postpro_code: int
            Error code of the postprocessing step
        update: bool
            True to update table file, False to append to table file
    """

    if update:
        update_status_table_file(status_table_file, run_index, prepro_code=prepro_code, run_code=run_code, 
                                 postpro_code=postpro_code)
    else:
        append_status_table_file(status_table_file, run_index, prepro_code=prepro_code, run_code=run_code, 
                                 postpro_code=postpro_code)


def process_runs(runs_file: str, index=None, resume: bool=False, **kwargs):
    """ Process runs

        Parameters
        ----------
        runs_file: str
            Path to the runs_file
        index: None, int or list
            Index or list of indices to runs or 
        resume: bool
            True to resume previous runs
        **kwags: dict
            Additionnal keyword arguments
    """

    logger = logging.getLogger("H2iVDI")

    if not os.path.isfile(runs_file):
        if not os.path.isabs(runs_file) and "input_dir" in kwargs:
            if os.path.isfile(os.path.join(kwargs["input_dir"], runs_file)):
                runs_file = os.path.join(kwargs["input_dir"], runs_file)

    # Open run_file
    runs_list = load_runs_file(runs_file)
    runs_indices = list(range(0, len(runs_list)))

    # Set update_status_table flag (True: update table, False: rewrite table)
    update_status_table = False
    if "update_status_table" in kwargs:
        if isinstance(update_status_table, bool):
            update_status_table = kwargs["update_status_table"]
        else:
            raise ValueError("'update_status_table' must be True or False")

    # Open log file if prescribed in kwargs
    status_table_fname = None
    if "status_table" in kwargs:
        if kwargs["status_table"] is not None:
            status_table_fname = kwargs["status_table"]
            if not os.path.isfile(status_table_fname) and not update_status_table:
                init_status_table_file(status_table_fname)
                # status_table_file = open(kwargs["status_table"], "w")
                # fcntl.flock(status_table_file, fcntl.LOCK_EX)
                # status_table_file.write("SET_INDEX;PREPRO;RUN;POSTPRO\n")
                # fcntl.flock(status_table_file, fcntl.LOCK_EX)
                # status_table_file.close()

    output_suffix = "hivdi"
    if index is not None:
        if isinstance(index, list):
            runs_indices = index
        else:
            runs_indices = [index]
        for index in runs_indices:
            if index < 0 or index >= len(runs_list):
                raise RuntimeError("Wrong run index: %i (must be in [0, %i[)" % (index, len(runs_list)))
        output_suffix = "hivdi"

    elif "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ:

        try:
            index = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
        except:
            raise RuntimeError("AWS_BATCH_JOB_ARRAY_INDEX must be a integer in the range [0, %i[" % len(runs_list))
        if index < 0 or index >= len(runs_list):
            raise RuntimeError("Wrong run index: %i (must be in [0, %i[)" % (index, len(runs_list)))
        runs_indices = [index]
        output_suffix = "h2ivdi"

    elif "EOHYDROLAB_SET_INDEX" in os.environ:

        try:
            index = int(os.environ["EOHYDROLAB_SET_INDEX"])
        except:
            raise RuntimeError("EOHYDROLAB_SET_INDEX must be a integer in the range [0, %i[" % len(runs_list))
        if index < 0 or index >= len(runs_list):
            raise RuntimeError("Wrong run index: %i (must be in [0, %i[)" % (index, len(runs_list)))
        runs_indices = [index]
        output_suffix = "hivdi"

    if "CONFLUENCE_US" in os.environ:
        output_suffix = "h2ivdi"
    #     kwargs["output_dir"] = "/mnt/data/flpe/hivdi"

    prepro_passed = 0
    run_passed = 0
    postpro_passed = 0
    critical_error_detected = False
    for index, run_index in enumerate(runs_indices):

        run_def = runs_list[run_index]

        logger.info("-" * 40)
        logger.info("Process set %i (%i/%i)" % (runs_indices[index], index+1, len(runs_indices)))

        if "data_type" in run_def:
            data_type = run_def["data_type"]
        else:
            data_type = "confluence"

        if data_type == "confluence":

            logger.info("Type: SWOT")
            if isinstance(run_def, list):
                logger.info("Reaches: %i - %i" % (int(run_def[0]["reach_id"]), int(run_def[-1]["reach_id"])))
            else:
                logger.info("Reach: %i" % int(run_def["reach_id"]))
            logger.info("-" * 40)

            # Check directories are present in kwargs
            if not "input_dir" in kwargs:
                raise ValueError("'input_dir' must be provided for confluence run")
            if not "output_dir" in kwargs:
                raise ValueError("'output_dir' must be provided for confluence run")

            # Create SwotCaseProcessor
            swot_options = kwargs["swot_options"]
            options = {"run-mode": swot_options["mode"],
                       "internal-data-correction": swot_options["internal-data-correction"]}
            processor = SwotCaseProcessor(run_def, options, kwargs["input_dir"], kwargs["output_dir"], kwargs["s3_path"])

        elif data_type == "pepsi":

            # Create and run PepsiCaseProcessor
            processor = PepsiCaseProcessor(os.path.expandvars(run_def["case_file"]))
            # processor.run()

        if resume is True:
            loaded = processor.resume(output_dir=kwargs["output_dir"])
            if loaded:
                logger.info("Skip already completed run")
                logger.info("-" * 40 + "\n")
                continue

        try:
            error_code = processor.prepro()
        except Exception as err:
            error_code = -9
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], -9, None, None,
                                                   update=update_status_table)
            # if status_table_file is not None:
                # status_table_file.write("%i;-9;;\n" % runs_indices[index])
            logger.exception("Preprocessing failed: %s" % repr(err))
            if logger._debug_level > 0:
                logger.error(traceback.format_exc())
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=997)
            critical_error_detected = True
            continue
        if error_code != 0:
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], error_code, None, None,
                                                   update=update_status_table)
            # if status_table_file is not None:
            #     status_table_file.write("%i;%i;;\n" % (runs_indices[index], error_code))
            logger.error("Preprocessing failed: error_code=%i" % error_code)
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=error_code)
            if error_code in __critical_errors__:
                critical_error_detected = True
            continue
        prepro_passed += 1
        
        try:
            error_code = processor.run()
        except Exception as err:
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], 0, -9, None,
                                                   update=update_status_table)
            # if status_table_file is not None:
            #     status_table_file.write("%i;0;-9;\n" % runs_indices[index])
            logger.error("Run failed: %s" % repr(err))
            if logger._debug_level > 0:
                logger.error(traceback.format_exc())
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=998)
            critical_error_detected = True
            continue
        if error_code != 0:
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], 0, error_code, None,
                                                   update=update_status_table)
            # if status_table_file is not None:
            #     status_table_file.write("%i;0;%i;\n" % (runs_indices[index], error_code))
            logger.error("Run failed: error_code=%i" % error_code)
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=error_code)
            if error_code in __critical_errors__:
                critical_error_detected = True
            continue
        run_passed += 1

        try:
            results, error_code = processor.postpro(output_dir=kwargs["output_dir"], suffix=output_suffix)
        except Exception as err:
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], 0, 0, -9,
                                                   update=update_status_table)
            # if status_table_file is not None:
            #     status_table_file.write("%i;0;0;-9\n" % runs_indices[index])
            logger.error("Postprocessing failed: %s" % repr(err))
            if logger._debug_level > 0:
                logger.error(traceback.format_exc())
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=999)
            critical_error_detected = True
        if error_code != 0:
            if status_table_fname is not None:
                append_or_update_status_table_file(status_table_fname, runs_indices[index], 0, 0, error_code,
                                                   update=update_status_table)
            # if status_table_file is not None:
            #     status_table_file.write("%i;0;0;%i\n" % (runs_indices[index], error_code))
            logger.error("Postprocessing failed: error_code=%i" % error_code)
            logger.info("-" * 40 + "\n")
            processor.write_failed_output(output_dir=kwargs["output_dir"], suffix=output_suffix, error_code=error_code)
            continue

        if status_table_fname is not None:
            append_or_update_status_table_file(status_table_fname, runs_indices[index], 0, 0, 0,
                                               update=update_status_table)
        # if status_table_file is not None:
        #     status_table_file.write("%i;0;0;0\n" % runs_indices[index])
        postpro_passed += 1
        logger.info("-" * 40 + "\n")


    #     if status_table_file is not None:
    #         status_table_file.flush()

    # if status_table_file is not None:
    #     status_table_file.close()

    logger.info("Execution status:")
    logger.info("- Preprocessing passed: %i/%i" % (prepro_passed, len(runs_list)))
    logger.info("- Run passed: %i/%i" % (run_passed, len(runs_list)))
    logger.info("- Postprocessing passed: %i/%i" % (postpro_passed, len(runs_list)))

    if critical_error_detected:
        raise RuntimeError("Critical error(s) detected")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("H2iVDI Command Line Interface (CLI)")
    parser.add_argument("runs_file", type=str,
                        help="Path to the run file")
    parser.add_argument("-i", dest="run_index",
                        type=int, help="Index of run in runs_file")
    parser.add_argument("-m", dest="run_mode",
                        type=str, default="unconstrained", choices=["constrained", "unconstrained"],
                        help="Run mode (SWOT)")
    parser.add_argument("--log-file", dest="log_file", type=str,
                        default=None,
                        help="Path to the log file")
    parser.add_argument("--input-dir", dest="input_dir", type=str,
                        default="/mnt/data/input",
                        help="Path to the input directory (for Confluence runs for instance)")
    parser.add_argument("--output-dir", dest="output_dir", type=str,
                        default="/mnt/data/output",
                        help="Path to the output directory (for Confluence runs for instance)")
    parser.add_argument("--status-table", dest="status_table", type=str,
                        default=None,
                        help="Path to the run status table file")
    parser.add_argument("--debug-level", dest="debug_level", type=int,
                        choices=[0, 1, 2], default=0,
                        help="Set debug level (0=debug disabled)")
    parser.add_argument("--disable-data-correction", dest="internal_data_correction", action="store_false",
                        help="Disable internal data corrections (SWOT)")
    parser.add_argument("--resume", action="store_true",
                        help="Enable resume mode")
    parser.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                        help="Retry failed runs (need option --status-table to be set)")
    parser.add_argument("--s3-path", dest="s3_path", type=str,
                        default='local',
                        help="Path for the SoS in S3 storage. Not providing the argument indicates local loading of the SoS")
    args = parser.parse_args()

    # Create loggger
    logger = create_logger(args.debug_level, args.log_file)

    if args.retry_failed:
        failed_indices = load_failed_runs_from_status_table_file(args.status_table)
        print(failed_indices)
        args.run_index = failed_indices


    # Perform runs
    process_runs(args.runs_file, index=args.run_index, input_dir=args.input_dir, output_dir=args.output_dir,
                 status_table=args.status_table, resume=args.resume, s3_path=args.s3_path, 
                 update_table=args.retry_failed, swot_options={"mode": args.run_mode, "internal-data-correction": args.internal_data_correction})

