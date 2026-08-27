import argparse
import time

def get_argument_parser():
    """Return the argument parser for histoqc"""
    parser = argparse.ArgumentParser(prog="histoqc", description='Run HistoQC main quality control pipeline for digital pathology images')
    parser.add_argument('input_pattern',
                        help="input filename pattern (try: *.svs or target_path/*.svs ),"
                             " or tsv file containing list of files to analyze",
                        nargs="+")
    parser.add_argument('-o', '--outdir',
                        help="outputdir, default ./histoqc_output_YYMMDD-hhmmss",
                        default=f"./histoqc_output_{time.strftime('%Y%m%d-%H%M%S')}",
                        type=str)
    parser.add_argument('-p', '--basepath',
                        help="base path to add to file names,"
                             " helps when producing data using existing output file as input",
                        default="",
                        type=str)
    parser.add_argument('-c', '--config',
                        help="config file to use, either by name supplied by histoqc.config (e.g., v2.1) or filename",
                        type=str)
    parser.add_argument('-f', '--force',
                        help="force overwriting of existing files",
                        action="store_true")
    parser.add_argument('-b', '--batch',
                        help="break results file into subsets of this size",
                        type=int,
                        default=None)
    parser.add_argument('-s', '--seed',
                        help="set a seed used to produce a random number in all modules",
                        type=int,
                        default=None)    
    parser.add_argument('-n', '--nprocesses',
                        help="number of processes to launch",
                        type=int,
                        default=1)
    parser.add_argument('--symlink', metavar="TARGET_DIR",
                        help="create symlink to outdir in TARGET_DIR",
                        default=None)

    parser.add_argument('--debug', action='store_true', help="trigger debugging behavior")

    return parser