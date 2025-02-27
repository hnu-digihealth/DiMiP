# Python Standard Library
from pathlib import Path

# Third Party Libraries

# Internal Libraries
from src.argparsers import data_setup_cli_parser, parse_path_arguments
from src.uncompress_data import unpack_cocahis, unpack_rings, unpack_and_tile_panda


# TODO add check if some files are already processed
# - do not rerun processing for finished datasets
# - rerun processing for partial datasets
# TODO add logging

def main(data_path: Path, remove_raw: bool):
    remove_raw = False # <- safety during development^^

    unpack_cocahis(data_path, remove_raw)
    unpack_rings(data_path, remove_raw)
    unpack_and_tile_panda(data_path, remove_raw)

    # check lbls 
    # '-> do we need to invert?
    # '-> do we need to parse to bitmap


if __name__ == '__main__':
    parser = data_setup_cli_parser()
    args = parser.parse_args()
    args = parse_path_arguments(args)
    main(**vars(args))