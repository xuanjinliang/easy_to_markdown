import argparse
import logging
import warnings
import sys
from easy_to_markdown.util.version import version
from easy_to_markdown.util.deprecation import CLIDeprecationWarning
from easy_to_markdown.util.logging import logger


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="easy_to_markdown",
        description="Easy to markdown command line tool",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {version}"
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    return parser


def _execute(args):
    args.executor(args)


def main():
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("default", category=CLIDeprecationWarning)
    parser = _get_parser()
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_usage(sys.stderr)
        sys.exit(2)
    _execute(args)


if __name__ == "__main__":
    main()
