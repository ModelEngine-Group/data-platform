import argparse
import os
import sys

from histoqc.data import package_resource_copytree


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="write package data to directory")
    parser.add_argument('out_dir',
                        help='Specify the data directory')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.out_dir):
        print(f"'{args.out_dir}' not a directory", file=sys.stderr)
        return -1

    # write histoqc package data to directory
    for rsrc in {'models', 'pen', 'templates'}:
        package_resource_copytree('histoqc.data', rsrc, args.out_dir, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
