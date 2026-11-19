import argparse
import logging
import warnings
import sys
from easy_to_markdown.util.version import version
from easy_to_markdown.util.deprecation import CLIDeprecationWarning, DeprecatedOptionAction
from easy_to_markdown.util.logging import logger
from easy_to_markdown.generate.run import cli_run_command


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="easy_to_markdown",
        description="Easy to markdown command line tool",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {version}", help="show version"
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND", help="Available commands")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert files",
    )

    convert_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration YAML file",
    )

    io_group = convert_parser.add_argument_group("Input / Output")

    io_group.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Input pdf path",
    )

    io_group.add_argument(
        "--temp-dir",
        default=None,
        help="temporary directory (default: current working directory)",
    )

    io_group.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: current working directory)",
    )

    pdf_group = convert_parser.add_argument_group("PDF Options")

    pdf_group.add_argument(
        "--pdf-dpi",
        type=int,
        default=None,
        help="PDF DPI (default: 200)",
    )

    pdf_group.add_argument(
        "--pdf-img-suffix",
        type=str,
        choices=["webp", "png"],
        default=None,
        help="PDF image format (webp or png)",
    )

    llm_group = convert_parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--model",
        default=None,
        help="LLM model name",
    )

    llm_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature",
    )

    llm_group.add_argument(
        "--base-url",
        default=None,
        help="LLM API base URL",
    )

    llm_group.add_argument(
        "--api-key",
        default=None,
        help="LLM API key",
    )

    convert_parser.set_defaults(executor=cli_run_command)

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
