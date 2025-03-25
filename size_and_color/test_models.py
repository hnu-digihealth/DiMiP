from os import cpu_count
from pathlib import Path
from src.ml_utils.test import test_model

from monai.utils.misc import set_determinism
from pytorch_lightning import seed_everything

from src.argparsers import data_setup_cli_parser, parse_path_arguments


def test_color_and_gray_models(data_path: Path, model_path: Path):
    cocahis_color_path = model_path / 'cocahis_color'
    cocahis_gray_path =  model_path / 'cocahis_grayscale'
    cocahis_data_path = data_path / 'processed' / 'cocahis'

    print('###################################')
    print('# -> Testing CoCaHis Color ########')
    print('###################################')
    cocahis_color_model = __get_ckpt_path_from_dir(cocahis_color_path) # get file with ckpt from model folder 
    test_model(
        test_image_path= cocahis_data_path / 'test',
        model_path=cocahis_color_model,
        batch_size = 12,
        num_workers= 4,
        normalizer_image_path= cocahis_data_path / 'test' / 'img' / 'patient-6_image-4.png',
        color_mode = 'color',
        visualization_path = cocahis_color_path
    )

    print('###################################')
    print('# -> Testing CoCaHis Gray #########')
    print('###################################')
    cocahis_gray_model = __get_ckpt_path_from_dir(cocahis_gray_path) # get file with ckpt from model folder 

    test_model(
        test_image_path= cocahis_data_path / 'test',
        model_path=cocahis_gray_model,
        batch_size = 12,
        num_workers= 4,
        normalizer_image_path= cocahis_data_path / 'test' / 'img' / 'patient-6_image-4.png',
        color_mode = 'gray',
        visualization_path = cocahis_gray_path
    )

    # print('###################################')
    # print('# -> Testing Rings Color ##########')
    # print('###################################')
    # train_and_validate_model(
    #     train_image_path= rings_path / 'train',
    #     val_image_path= rings_path / 'validate',
    #     batch_size = 12,
    #     mode='gpu',
    #     model_path= Path('./rings_color'),
    #     lbl_folder='lbl_tumor',
    #     num_workers= 4,
    #     normalizer_image_path= rings_path / 'test' / 'img' / 'P3_C11_13_8_5.png',
    #     color_mode = 'color',
    # )

    # print('###################################')
    # print('# -> Testing Rings Gray ###########')
    # print('###################################')
    # train_and_validate_model(
    #     train_image_path= rings_path / 'train',
    #     val_image_path= rings_path / 'validate',
    #     batch_size = 12,
    #     mode='gpu',
    #     model_path= Path('./rings_grayscale'),
    #     lbl_folder='lbl_tumor',
    #     num_workers= 4,
    #     color_mode = 'gray',
    # )

    # print('###################################')
    # print('# -> Testing Panda Color ##########')
    # print('###################################')
    # train_and_validate_model(
    #     train_image_path= panda_path / 'train',
    #     val_image_path= panda_path / 'validate',
    #     batch_size = 64,
    #     mode='gpu',
    #     model_path= Path('./panda_color'),
    #     num_workers= 30,
    #     lbl_folder='lbl_5x',
    #     img_folder='img_5x',
    #     normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
    #     color_mode = 'color',
    #     classes=6
    # )

    # print('###################################')
    # print('# -> testing Panda Gray ###########')
    # print('###################################')
    # train_and_validate_model(
    #     train_image_path= panda_path / 'train',
    #     val_image_path= panda_path / 'validate',
    #     batch_size = 64,
    #     mode='gpu',
    #     model_path= Path('./panda_grayscale'),
    #     num_workers= 30,
    #     lbl_folder='lbl_5x',
    #     img_folder='img_5x',
    #     color_mode = 'gray',
    #     classes=6
    # )


# def test_size_models(data_path: Path):
#     panda_path = data_path / 'processed' / 'panda'

#     # panda 5x -> size and color model

#     print('###################################')
#     print('# -> Testing Panda Color ##########')
#     print('###################################')
#     train_and_validate_model(
#         train_image_path= panda_path / 'train',
#         val_image_path= panda_path / 'validate',
#         batch_size = 64,
#         mode='gpu',
#         model_path= Path('./panda_10x'),
#         num_workers= 30,
#         lbl_folder='lbl_10x',
#         img_folder='img_10x',
#         normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
#         color_mode = 'color',
#         classes=6
#     )

#     print('###################################')
#     print('# -> Testing Panda Color ##########')
#     print('###################################')
#     train_and_validate_model(
#         train_image_path= panda_path / 'train',
#         val_image_path= panda_path / 'validate',
#         batch_size = 64,
#         mode='gpu',
#         model_path= Path('./panda_20x'),
#         num_workers= 30,
#         lbl_folder='lbl_20x',
#         img_folder='img_20x',
#         normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
#         color_mode = 'color',
#         classes=6
#     )


def __get_ckpt_path_from_dir(ckpt_dir: Path) -> Path:
    """
    Get the path to the .ckpt file from a given directory.
    Assumes there is exactly one .ckpt file in the directory.

    Args:
        ckpt_dir (Path): The directory containing the checkpoint.

    Returns:
        Path: Path to the .ckpt file.

    Raises:
        FileNotFoundError: If no .ckpt file is found.
        RuntimeError: If multiple .ckpt files are found.
    """
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No .ckpt file found in directory: {ckpt_dir}")
    elif len(ckpt_files) > 1:
        raise RuntimeError(f"Multiple .ckpt files found in directory: {ckpt_dir}. Expected only one.")

    return ckpt_files[0]


if __name__ == '__main__':
    parser = data_setup_cli_parser()
    args = parser.parse_args()
    args = parse_path_arguments(args)

    set_determinism(seed=421337133742)
    seed_everything(421337133742, workers=True)

    test_color_and_gray_models(args.data_path, args.model_path)
    # test_size_models(args.data_path)
