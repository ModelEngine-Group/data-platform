import argparse
import configparser
import datetime
import glob
import logging
import multiprocessing
import os
import sys
import time
from functools import partial

from histoqc._pipeline import BatchedResultFile
from histoqc._pipeline import MultiProcessingLogManager
from histoqc._pipeline import load_pipeline
from histoqc._pipeline import log_pipeline
from histoqc._pipeline import move_logging_file_handler
from histoqc._pipeline import setup_logging
from histoqc._pipeline import setup_plotting_backend
from histoqc._worker import worker
from histoqc._worker import worker_setup
from histoqc._worker import worker_success
from histoqc._worker import worker_error
from histoqc.config import read_config_template
from histoqc.data import managed_pkg_data
from histoqc._parser import get_argument_parser

parser = get_argument_parser()

@managed_pkg_data
def main(argv=None):
    """main entry point for histoqc pipelines"""
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    # --- multiprocessing and logging setup -----------------------------------

    setup_logging(capture_warnings=True, filter_warnings='ignore')
    mpm = multiprocessing.Manager()
    lm = MultiProcessingLogManager('histoqc', manager=mpm)

    # --- parse the pipeline configuration ------------------------------------
    config = configparser.ConfigParser()
    if not args.config:
        lm.logger.warning(f"Configuration file not set (--config), using default")
        config.read_string(read_config_template('default'))
    elif os.path.exists(args.config):
        config.read(args.config) #Will read the config file
    else:
        lm.logger.warning(f"Configuration file {args.config} assuming to be a template...checking.")
        config.read_string(read_config_template(args.config))

    # --- provide models, pen and templates as fallbacks from package data ----

    managed_pkg_data.inject_pkg_data_fallback(config)

    # --- load the process queue (error early) --------------------------------

    _steps = log_pipeline(config, log_manager=lm)
    process_queue = load_pipeline(config)

    # --- check symlink target ------------------------------------------------

    if args.symlink is not None:
        if not os.path.isdir(args.symlink):
            lm.logger.error("error: --symlink {args.symlink} is not a directory")
            return -1

    # --- create output directory and move log --------------------------------
    args.outdir = os.path.expanduser(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    move_logging_file_handler(logging.getLogger(), args.outdir, args.debug)

    if BatchedResultFile.results_in_path(args.outdir):
        if args.force:
            lm.logger.info("Previous run detected....overwriting (--force set)")
        else:
            lm.logger.info("Previous run detected....skipping completed (--force not set)")

    results = BatchedResultFile(args.outdir,
                                manager=mpm,
                                batch_size=args.batch,
                                force_overwrite=args.force)

    # --- document configuration in results -----------------------------------

    results.add_header(f"start_time:\t{datetime.datetime.now()}")
    results.add_header(f"pipeline:\t{' '.join(_steps)}")
    results.add_header(f"outdir:\t{os.path.realpath(args.outdir)}")
    results.add_header(f"config_file:\t{os.path.realpath(args.config) if args.config is not None else 'default'}")
    results.add_header(f"command_line_args:\t{' '.join(argv)}")

    # --- receive input file list (there are 3 options) -----------------------
    args.basepath = os.path.expanduser(args.basepath)
    if len(args.input_pattern) > 1:
        # more than one input_pattern is interpreted as a list of files
        # (basepath is ignored)
        files = list(args.input_pattern)

    elif args.input_pattern[0].endswith('.tsv'):
        # input_pattern is a tsv file containing a list of files
        files = []
        with open(args.input_pattern[0], 'rt') as f:
            for line in f:
                if line[0] == "#":
                    continue
                fn = line.strip().split("\t")[0]
                files.append(os.path.join(args.basepath, fn))

    else:
        # input_pattern is a glob pattern
        pth = os.path.join(args.basepath, args.input_pattern[0])
        files = glob.glob(pth, recursive=True)

    lm.logger.info("-" * 80)
    num_files = len(files)
    lm.logger.info(f"Number of files detected by pattern:\t{num_files}")

    # --- start worker processes ----------------------------------------------

    _shared_state = {
        'process_queue': process_queue,
        'config': config,
        'outdir': args.outdir,
        'log_manager': lm,
        'lock': mpm.Lock(),
        'shared_dict': mpm.dict(),
        'num_files': num_files,
        'force': args.force,
        'seed': args.seed,
        'debug': args.debug
    }
    failed = mpm.list()
    setup_plotting_backend(lm.logger, debug=args.debug)

    try:
        if args.nprocesses > 1:

            with lm.logger_thread():
                print(args.nprocesses)
                with multiprocessing.Pool(processes=args.nprocesses,
                                          initializer=worker_setup,
                                          initargs=(config,)) as pool:
                    try:
                        for idx, file_name in enumerate(files):
                            _ = pool.apply_async(
                                func=worker,
                                args=(idx, file_name),
                                kwds=_shared_state,
                                callback=partial(worker_success, result_file=results),
                                error_callback=partial(worker_error, failed=failed),
                            )

                    finally:
                        pool.close()
                        pool.join()

        else:
            for idx, file_name in enumerate(files):
                try:
                    _success = worker(idx, file_name, **_shared_state)
                except Exception as exc:
                    worker_error(exc, failed)
                    continue
                else:
                    worker_success(_success, results)

    except KeyboardInterrupt:
        lm.logger.info("-----REQUESTED-ABORT-----\n")

    else:
        lm.logger.info("----------Done-----------\n")

    finally:
        lm.logger.info(f"There are {len(failed)} explicitly failed images (available also in error.log),"
                       " warnings are listed in warnings column in output")

        for file_name, error, tb in failed:
            lm.logger.info(f"{file_name}\t{error}\n{tb}")
    timing_file = os.path.join(args.outdir, "processing_times.tsv")
    file_exists = os.path.exists(timing_file)
    with open(timing_file, 'a') as tf:
        if not file_exists:
            tf.write("slide_name\tfile_size_mb\tprocessing_time_sec\n")

        for file_name in files:
            base_name = os.path.basename(file_name)
            file_size_bytes = os.path.getsize(file_name) if os.path.exists(file_name) else 0
            file_size_mb = file_size_bytes / (1024 * 1024)
           
            time_key = f"time_{base_name}"
            time_taken = _shared_state['shared_dict'].get(time_key, 0.0)
           
            tf.write(f"{base_name}\t{file_size_mb:.2f}\t{time_taken:.2f}\n")
    lm.logger.info(f"Processing times appended to {timing_file}")
    
    if args.symlink is not None:
        origin = os.path.realpath(args.outdir)
        target = os.path.join(
            os.path.realpath(args.symlink),
            os.path.basename(origin)
        )
        try:
            os.symlink(origin, target, target_is_directory=True)
            lm.logger.info("Symlink to output directory created")
        except (FileExistsError, FileNotFoundError):
            lm.logger.error(
                f"Error creating symlink to output in '{args.symlink}', "
                f"Please create manually: ln -s {origin} {target}"
            )
    return 0

if __name__ == "__main__":
    sys.exit(main())
