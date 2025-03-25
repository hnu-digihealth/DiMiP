from os import cpu_count
from pathlib import Path
from src.ml_utils.train import train_and_validate_model

from src.argparsers import data_setup_cli_parser, parse_path_arguments

def train_color_and_gray_models(data_path: Path):
    cocahis_path = data_path / 'processed' / 'cocahis'
    rings_path = data_path / 'processed' / 'rings'
    panda_path = data_path / 'processed' / 'panda'

    print('###################################')
    print('# -> Training CoCaHis Color #######')
    print('###################################')
    train_and_validate_model(
        train_image_path= cocahis_path / 'train',
        val_image_path= cocahis_path / 'validate',
        batch_size = 12,
        mode= 'gpu',
        model_path= Path('./cocahis_color'),
        num_workers= 4,
        normalizer_image_path= cocahis_path / 'test' / 'img' / 'patient-6_image-4.png',
        color_mode = 'color',
    )
    print('###################################')
    print('# -> Training CoCaHis Gray ########')
    print('###################################')

    train_and_validate_model(
        train_image_path= cocahis_path / 'train',
        val_image_path= cocahis_path / 'validate',
        batch_size = 12,
        mode= 'gpu',
        model_path= Path('./cocahis_grayscale'),
        num_workers= 4,
        color_mode = 'gray',
    )

    print('###################################')
    print('# -> Training Rings Color #########')
    print('###################################')
    train_and_validate_model(
        train_image_path= rings_path / 'train',
        val_image_path= rings_path / 'validate',
        batch_size = 12,
        mode='gpu',
        model_path= Path('./rings_color'),
        lbl_folder='lbl_tumor',
        num_workers= 4,
        normalizer_image_path= rings_path / 'test' / 'img' / 'P3_C11_13_8_5.png',
        color_mode = 'color',
    )

    print('###################################')
    print('# -> Training Rings Gray ##########')
    print('###################################')
    train_and_validate_model(
        train_image_path= rings_path / 'train',
        val_image_path= rings_path / 'validate',
        batch_size = 12,
        mode='gpu',
        model_path= Path('./rings_grayscale'),
        lbl_folder='lbl_tumor',
        num_workers= 4,
        color_mode = 'gray',
    )

    print('###################################')
    print('# -> Training Panda Color #########')
    print('###################################')
    train_and_validate_model(
        train_image_path= panda_path / 'train',
        val_image_path= panda_path / 'validate',
        batch_size = 64,
        mode='gpu',
        model_path= Path('./panda_color'),
        num_workers= 30,
        lbl_folder='lbl_5x',
        img_folder='img_5x',
        normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
        color_mode = 'color',
        classes=6
    )

    print('###################################')
    print('# -> Training Panda Gray ##########')
    print('###################################')
    train_and_validate_model(
        train_image_path= panda_path / 'train',
        val_image_path= panda_path / 'validate',
        batch_size = 64,
        mode='gpu',
        model_path= Path('./panda_grayscale'),
        num_workers= 30,
        lbl_folder='lbl_5x',
        img_folder='img_5x',
        color_mode = 'gray',
        classes=6
    )


def train_size_models(data_path: Path):
    panda_path = data_path / 'processed' / 'panda'

    # panda 5x -> size and color model

    print('###################################')
    print('# -> Training Panda Color #########')
    print('###################################')
    train_and_validate_model(
        train_image_path= panda_path / 'train',
        val_image_path= panda_path / 'validate',
        batch_size = 64,
        mode='gpu',
        model_path= Path('./panda_10x'),
        num_workers= 30,
        lbl_folder='lbl_10x',
        img_folder='img_10x',
        normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
        color_mode = 'color',
        classes=6
    )

    print('###################################')
    print('# -> Training Panda Color #########')
    print('###################################')
    train_and_validate_model(
        train_image_path= panda_path / 'train',
        val_image_path= panda_path / 'validate',
        batch_size = 64,
        mode='gpu',
        model_path= Path('./panda_20x'),
        num_workers= 30,
        lbl_folder='lbl_20x',
        img_folder='img_20x',
        normalizer_image_path= panda_path / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png',
        color_mode = 'color',
        classes=6
    )


if __name__ == '__main__':
    parser = data_setup_cli_parser()
    args = parser.parse_args()
    args = parse_path_arguments(args)
    
    train_color_and_gray_models(args.data_path)
    # train_size_models(args.data_path)