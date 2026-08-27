import argparse

def get_argument_parser():
    """Return the argument parser for histoqc.config"""
    parser = argparse.ArgumentParser(prog="histoqc.config",description="Show example configuration files")
    parser.add_argument('--list',
                        action='store_true',
                        help='list available configs')
    parser.add_argument('--show',
                        metavar='NAME',
                        type=str,
                        default=None,
                        help='show named example config')
    return parser
