# Python Standard Library
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Third Party Libraries

# Internal Libraries


#-----------------------------------------------------#
#                 Data Setup - Parser                 #
#-----------------------------------------------------#
def data_setup_cli_parser():
    """
    CLI argument parser used by the setup_data script.

    Returns
    ----------
    - `parser: ArgumentParser` - The argument parser for the setup_data script.
    """
    desc = 'Setup utility to prepare the public datasets for usage with the MONAI ML code in this project.'

    parser = ArgumentParser(prog='data_setup', description=desc)

    ra = parser.add_argument_group('Required Arguments')

    oa = parser.add_argument_group('Optional Arguments')       
    oa.add_argument(
        '-dp', '--data_path',
        type=Path,
        required=False,
        default='data',
        help='Path to the top level data directory. This directory will contain all the raw and preprocessed data.'
    )

    oa.add_argument(
        '-mp', '--model_path',
        type=Path,
        required=False,
        default='models',
        help='Path to the directory where the trained models are.'
    )

    oa.add_argument(
        '-rr', '--remove_raw',
        action='store_true',
        help='Flag if the raw data should be removed after processing to save disc space.'
    )

    oa.add_argument(
        '-sv', '--save_visualization',
        action='store_true',
        help='Flag if the visualization of the data should be saved.'
    )

    oa.add_argument(
    '-rp', '--results_path',
        type=Path,
        required=False,
        default='results',
        help='Path to the directory where the results are saved.'
    )

    # Return parsers
    return parser


#-----------------------------------------------------#
#                  Helper Functions                   #
#-----------------------------------------------------#
def parse_path_arguments(args: Namespace) -> Namespace:
    """
    Internal helper function parsing path arguments in the   provided config to Python `Path` objects.

    Parameters
    ----------
    - `args: Namespace` - The parsed arguments from the CLI.

    Returns
    ----------
    - `args: Namespace` - The updated arguments with the path arguments converted to `Path` objects.
    """
    for key, value in vars(args).items():
        if key.endswith('_path'):
            setattr(args, key, Path(value))
    return args