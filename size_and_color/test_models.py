from os import cpu_count
from pathlib import Path
from src.ml_utils.test import test_model

from monai.utils.misc import set_determinism
from torch import set_float32_matmul_precision
import pandas as pd

from src.argparsers import data_setup_cli_parser, parse_path_arguments
from src.analysis_utils.metrics import merge_by_epochs, get_best_epochs

def test_models(
    model_dir: Path,
    data_dir: Path,
    visualize: bool = False,
    results_dir: Path = None
) -> None:
    """
    Aggregates all models from `model_dir`. The `model_dir` should contain the top level folders with the dataset names
    and a identifier (e.g., cocahis_grayscale or pandas_5x). Based on the name the right test dataset is selected from the
    top  level data directory `data_dir`. Afterwards the best metrics (loss, dice) are extracted from the models metric
    folder and the best_loss and best_dice checkpoint are used for training.

    The test results are merged with the validation results and are saved as a single csv for all models to 
    `results_dir`. All test scores are further saved image wise for the specific models to an own csv file.

    If `visualize` is set to True, the 3 best and worst test results are saved with ground truth and prediction.
    
    Parameters
    ----------
    - model_dir: Path
        The top level directory containing the model folders for cocahis, rings, panda.
    - data_dir: Path
        The top level directory containing the processed dataset folders for cocahis, rings, panda.
    - visualize: bool
        If True, the 3 best and worst test results are saved with ground truth and prediction.
    - results_dir: Path
        If None, the results are saved to `model_dir`/results. Otherwise, the results are saved to `results_dir`.
    
    Returns
    -------
    - None
    """

    models = get_model_data_from_dir(model_dir)
    if results_dir is None:
        results_dir = Path('.') / 'results'

    for model in models:
        results = []
        print('#################################################')
        print(f' -> Testing model: {model["id"]}_{model["type"]}')
        print('#################################################')
        # run test on model and save all dice results as well as images
        color_mode = 'color' if model['type'] != 'grayscale' else 'grayscale'
        normalizer_image_path = Path()
        if model['id'] == 'cocahis':
            test_image_path = data_dir / 'cocahis' / 'test' 
            normalizer_image_path = data_dir / 'cocahis' / 'test' / 'img' / 'patient-6_image-4.png'
            lbl_folder='lbl'
            img_folder='img'
            classes = 1
            visualizeation_path = results_dir / f'cocahis_{model["type"]}'
            visualizeation_path.mkdir(parents=True, exist_ok=True)
            batch_size = 12
        elif model['id'] == 'rings':
            test_image_path = data_dir / 'rings' / 'test'
            normalizer_image_path = data_dir / 'rings' / 'test' / 'img' / 'P3_C11_13_8_5.png'
            lbl_folder='lbl_tumor'
            img_folder='img'
            classes = 1
            visualizeation_path = results_dir / f'rings_{model["type"]}'
            visualizeation_path.mkdir(parents=True, exist_ok=True)
            batch_size = 18
        elif model['id'] == 'panda':
            test_image_path = data_dir / 'panda' / 'test'
            normalizer_image_path = data_dir / 'panda' / 'train' / 'img_20x' / '01a9472f2b061f80bb7c125dfa9771e5_row2048_col4096.png'
            classes = 6
            visualizeation_path = results_dir / f'panda_{model["type"]}'
            visualizeation_path.mkdir(parents=True, exist_ok=True)
            batch_size = 32
            if model['type'] == '5x' or model['type'] == 'grayscale':
                lbl_folder='lbl_5x'
                img_folder='img_5x'
            else:
                lbl_folder='lbl_10x'
                img_folder='img_10x'

        for model_path in [model['best_loss'], model['best_dice']]:
            curr_visualizeation_path = visualizeation_path / model_path.name
            curr_visualizeation_path.mkdir(parents=True, exist_ok=True)
            results.append(test_model(
                test_image_path=test_image_path,
                normalizer_image_path=normalizer_image_path,
                model_path=model_path,
                batch_size=batch_size,
                num_workers=16,
                img_folder=img_folder,
                lbl_folder=lbl_folder,
                color_mode=color_mode,
                classes=classes,
                visualization_path=curr_visualizeation_path if visualize else None
            ))

        print(results)
        # aggregate test metrics and validation metrics
        metric = pd.read_csv(model['metrics'])
        metric = merge_by_epochs(metric)
        best_epochs = get_best_epochs(metric)
        best_df = pd.DataFrame([best_epochs['best_dice'], best_epochs['best_loss']], index=["best_dice_val", "best_loss_val"])
        test = pd.DataFrame([results[0], results[1]], index=["best_dice_test", "best_loss_test"])
        
        best_df = pd.concat([best_df, test], sort=False)

        best_df.to_csv(results_dir / f'{model["id"]}_{model["type"]}_best_metrics.csv', index=True)


def get_model_data_from_dir(model_dir: Path):
    models = []
    # id: ,type, metrics, best_loss, best_dice
    for folder in model_dir.iterdir():
        if folder.is_dir():
            try:
                name = folder.name.split('_')[0]
            except:
                continue
            if name in ['cocahis', 'rings', 'panda']:
                model = {
                    'id': folder.name.split('_')[0],
                    'type': folder.name.split('_')[1],
                    'metrics': folder / 'lightning_logs' / 'version_0' / 'metrics.csv',
                    'best_loss': folder / 'lightning_logs' / 'version_0' / 'checkpoints' / 'best_loss.ckpt',
                    'best_dice': folder / 'lightning_logs' / 'version_0' / 'checkpoints' / 'best_dice.ckpt',
                }
                models.append(model)

    return models


if __name__ == '__main__':
    parser = data_setup_cli_parser()
    args = parser.parse_args()
    args = parse_path_arguments(args)

    set_determinism(seed=421337133742)
    set_float32_matmul_precision('medium')

    test_models(args.model_path, args.data_path, args.save_visualization, args.results_path)
