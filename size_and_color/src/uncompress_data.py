# TODO impl progress bar for uncompressing
# >>> with zipfile.ZipFile(some_source) as zf:
#     for member in tqdm(zf.infolist(), desc='Extracting '):
#         try:
#             zf.extract(member, target_path)
#         except zipfile.error as e:
#             pass
# TODO fix cleanup of logic (does currently not delete all files)

# Python Standard Library
from pathlib import Path
from shutil import copy
import zipfile
from itertools import product
from multiprocessing import Pool

# Third Party Libraries
import h5py
import pandas as pd
from openslide import OpenSlide, OpenSlideError
from PIL import Image
from matplotlib.image import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Internal Libraries


# TODO check if all images have same resolution
# else rescale iamges to 1024x1024
# we will require a padding aproach for this as images are not square
def unpack_cocahis(dir: Path, remove_raw: bool) -> None:
    """
    Function to extract the CoCaHis data from the provided directory containing the raw hdf5 dataset and save it in the
    corresponding processed folder.

    Parameters
    ----------
    - `dir: Path` - The top level data directory containing the raw CoCaHis hdf5 dataset.
    - `remove_raw: bool` - Flag if the raw data should be removed after processing to save disc space.

    Returns
    ----------
    - `None`
    """
    raw_dir = dir / 'raw' / 'cocahis'
    processed_dir = dir / 'processed' / 'cocahis'

    with h5py.File(raw_dir / 'CoCaHis.hdf5', 'r') as f:
        raw_images = f["HE/raw"][()]
        gt = f["GT/GT_majority_vote"][()]
    
        trtst = f["HE/"].attrs["train_test_split"]
        patients = f["HE/"].attrs["patient_num"]
        img_num = f["HE/"].attrs["image_num"]

    # Dataset has only train/test split 
    # patient 4 and 19 will be used for validation
    # as test split is quite large and train small move patient 11 to train
    for i in range(len(patients)):
        if patients[i] in ['4', '19']:
            trtst[i] = 'validate'
        elif patients[i] == '11':
            trtst[i] = 'train'

    # generate folder structure in processed folder
    __init_ml_folder_structure(processed_dir)

    for i in tqdm(range(len(raw_images)), desc='Extracting CoCaHis dataset'):
        img = raw_images[i]
        lbl = gt[i]
        trtst_ = trtst[i]
        patient = patients[i]
        patient_img_num = img_num[i]

        # save image to respective folder
        img = Image.fromarray(img)
        __add_bottom_margin(img, 351, (255,255,255)).resize((1024,1024)).save(processed_dir / trtst_ / 'img' / f'patient-{patient}_image-{patient_img_num}.png')
        lbl = Image.fromarray(lbl)
        __add_bottom_margin(lbl, 351, 0).resize((1024,1024)).save(processed_dir / trtst_ / 'lbl' / f'patient-{patient}_image-{patient_img_num}.png')

    if remove_raw:
        print('Removing raw CoCaHis data')
        (raw_dir / 'CoCaHis.hdf5').unlink()


def unpack_rings(dir: Path, remove_raw: bool) -> None:
    """
    Function to extract the RINGS data from the provided directory containing the raw zipped dataset and save it in the
    corresponding processed folder.

    Parameters
    ----------
    - `dir: Path` - The top level data directory containing the raw zipped RINGS dataset.
    - `remove_raw: bool` - Flag if the raw data should be removed after processing to save disc space.

    Returns
    ----------
    - `None`
    """
    raw_dir = dir / 'raw' / 'rings'
    processed_dir = dir / 'processed' / 'rings'

    # unzip rings data and nested train/test data
    with zipfile.ZipFile(raw_dir / 'RINGS algorithm dataset.zip', 'r') as zip_ref:
        print('Extracting RINGS dataset')
        zip_ref.extractall(raw_dir)

    with zipfile.ZipFile(raw_dir / 'RINGS algorithm dataset/TRAIN.zip', 'r') as zip_ref:
        print(" '-> Extracting TRAIN data")
        zip_ref.extractall(raw_dir / 'RINGS algorithm dataset')

    with zipfile.ZipFile(raw_dir / 'RINGS algorithm dataset/TEST.zip', 'r') as zip_ref:
        print(" '-> Extracting TEST data")
        zip_ref.extractall(raw_dir / 'RINGS algorithm dataset')

    # generate folder structure in processed folder
    __init_ml_folder_structure(processed_dir, 'rings')

    # move images to respective folders
    # regenerate split into 1200/200/100 split to increase data for training using test data
    for folder in ['TRAIN', 'TEST']:
        img_path = raw_dir / 'RINGS algorithm dataset' / folder / 'IMAGES'
        lbl_tumor = raw_dir / 'RINGS algorithm dataset' / folder / 'MANUAL TUMOR'
        lbl_gland = raw_dir / 'RINGS algorithm dataset' / folder / 'MANUAL GLANDS'
        
        if folder == 'TRAIN':
            folder = 'train'

            for img in tqdm([img for img in img_path.iterdir() if not img.name.startswith('.')], desc='Copying RINGS TRAIN data'):   
                Image.open(
                    img
                ).resize((1024, 1024)).save(processed_dir / folder / 'img' / img.name)
                Image.fromarray(imread(
                    lbl_tumor / img.name
                )).convert("L").resize((1024, 1024)).save(processed_dir / folder / 'lbl_tumor' / img.name)
                Image.fromarray(imread(
                    lbl_gland / img.name
                )).convert("L").resize((1024, 1024)).save(processed_dir / folder / 'lbl_gland' / img.name)
        else:
            __resplit_rings_test_data(img_path, lbl_tumor, lbl_gland, processed_dir)  

    if remove_raw:
        print('Removing raw RINGS data')
        (raw_dir / 'RINGS algorithm dataset.zip').unlink()
        (raw_dir / 'RINGS algorithm dataset').rmdir()


def unpack_and_tile_panda(dir: Path, remove_raw: bool) -> None:
    """
    Function to extract the PANDA data from the provided directory containing the raw zipped dataset and save it in the
    corresponding processed folder.

    Only data from radboud is used, as the data from karolinska has different/less precise labels.

    Parameters
    ----------
    - `dir: Path` - The top level data directory containing the raw zipped PANDA dataset.
    - `remove_raw: bool` - Flag if the raw data should be removed after processing to save disc space.

    Returns
    ----------
    - `None`
    """
    raw_dir = dir / 'raw' / 'panda'
    processed_dir = dir / 'processed' / 'panda'

    # unzip panda data
    # with zipfile.ZipFile(raw_dir / 'prostate-cancer-grade-assessment.zip', 'r') as zip_ref:
    #     print('Extracting PANDA dataset')
    #     zip_ref.extractall(raw_dir)

    # generate folder structure in processed folder
    __init_ml_folder_structure(processed_dir, 'panda')

    images = pd.read_csv(raw_dir / 'train.csv')
    images = images[images['data_provider'] == 'radboud']
    tiles_used = {'test': [], 'validate': [], 'train': []}
    tiles_used_counter = 0
    tasks = []

    # generate tasks with according folder allocation
    for img_id in tqdm(images['image_id'],
        desc='Generating tiling tasks for dataset processing'                   
    ):
        if tiles_used_counter < 10:
            folder = 'test'
        elif tiles_used_counter < 30:
            folder = 'validate'
        else:
            folder = 'train'

        tiles_used[folder].append(img_id)
        save_path = processed_dir / folder
        tasks.append((img_id, raw_dir, save_path))
        tiles_used_counter += 1

    # iterate over tain_label_masks (as we have less masks than images)
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap_unordered(__tile_worker, tasks), total=len(tasks), desc="Tiling PANDA dataset"))
    
    # save the used tiles in a csv file
    # TODO fix this
    #pd.DataFrame(tiles_used).to_csv(processed_dir / 'tiles_used.csv')

    if remove_raw:
        print('Removing raw PANDA data')
        (raw_dir / 'panda.zip').unlink()
        (raw_dir / 'panda').rmdir()


def __tile_worker(task):
    img_id, raw_dir, save_path = task
    try:
        label = OpenSlide(raw_dir / 'train_label_masks' / (img_id + '_mask.tiff'))
        slide = OpenSlide(raw_dir / 'train_images' / (img_id + '.tiff'))
        __extract_tiles(slide, label, save_path, img_id)
    except OpenSlideError:
        return None
    except Exception as e:
        print(f"Unexpected error for {img_id}: {e}")
    finally:
        try:
            slide.close()
            label.close()
        except:
            pass
    return img_id

def __resplit_rings_test_data(
    img_path: Path,
    lbl_tumor: Path,
    lbl_gland: Path,
    processed_dir: Path
) -> None:
    """
    Helper function to generate a 1200/200/100 split for the RINGS test data. as the 1000/500 split is not optimal for
    ML. The split tries to consider patients, due to suboptimal documentation this is however not guaranteed.<br>
    The patient groupings were manually determined from the provided data.<br>
    All images are rescaled from 1500x1500px to 1024x1024px resolution.

    Parameters
    ----------
    - `img_path: Path` - The path to the images of the RINGS test data.
    - `lbl_tumor: Path` - The path to the tumor labels of the RINGS test data.
    - `lbl_gland: Path` - The path to the gland labels of the RINGS test data.
    - `processed_dir: Path` - The top level processed directory to save the data in.

    Returns
    ----------
    - `None`
    """
    for img in tqdm([img for img in img_path.iterdir() if not img.name.startswith('.')], desc='Resplitting RINGS TEST data'):      
        folder = 'validate'
        if '_'.join(img.name.split('_')[:2]) in [ 'P7_A8', 'P3_C11', 'P4_C9',
            'P6_E5', 'P4_C5', 'P5_D8', 'P3_C9', 'P1_D3', 'P2_F3', 'P1_D5', 'P3_A13',
            'P1_D6', 'P5_D4', 'P8_E4', 'P7_A3', 'P3_A5', 'P4_C4', 'P4_A12', 'P4_C10'
        ]:
            folder = 'test'
        elif '_'.join(img.name.split('_')[:2]) in [ 'P6_E7', 'P5_D5', 'P3_A11',
            'P4_A5', 'P5_D7', 'P4_C11', 'P1_D9','P8_E3', 'P5_D3', 'P2_F7', 'P1_D4', 'P1_D10'
        ]:
            folder = 'train'

        Image.open(img).resize((1024, 1024)).save(processed_dir / folder / 'img' / img.name)
        Image.fromarray(imread(
            lbl_tumor / img.name
        )).convert("L").resize((1024, 1024)).save(processed_dir / folder / 'lbl_tumor' / img.name)
        Image.fromarray(imread(
            lbl_gland / img.name
        )).convert("L").resize((1024, 1024)).save(processed_dir / folder / 'lbl_gland' / img.name)


def __init_ml_folder_structure(data_dir: Path, dataset: str | None = None) -> None:
    """
    Helper function generating a test/train/valdiate folder structure in the provided data directory.

    Parameters
    ----------
    - `data_dir: Path` - The top level data directory to generate the folder structure in.

    Returns
    ----------
    - `None`
    """
    subfolders = ['img', 'lbl']
    if dataset == 'rings':
        subfolders = ['img', 'lbl_tumor', 'lbl_gland']
    elif dataset == 'panda':
        subfolders = ['_'.join(sub) for sub in product(['img', 'lbl'], ['5x', '10x', '20x'])]

    for folder in ['train', 'test', 'validate']:
        (data_dir / folder).mkdir(parents=True, exist_ok=True)
        for subfolder in subfolders:
            (data_dir / folder / subfolder).mkdir(parents=True, exist_ok=True)


def __extract_tiles(slide: OpenSlide, label: OpenSlide, save_path: Path, id: str) -> None:
    """
    Extract tiles from the panda dataset slides at 5x, 10x and 20x magnification. The tiles are saved in the respective
    folders. The tiles are only saved if the label contains more than 15% of non-background pixels.

    Parameters
    ----------
    - `slide: OpenSlide` - The slide to extract the tiles from.
    - `label: OpenSlide` - The label slide to extract the tiles from.
    - `save_path: Path` - The top level directory to save the tiles in.
    - `id: str` - The id of the slide.

    Returns
    ----------
    - `None`
    """
    tile_size = 1024

    # 5x
    for y in range(0, label.level_dimensions[1][1], tile_size):
        for x in range(0, label.level_dimensions[1][0], tile_size):
            try:
                lbl_tile = label.read_region((x*4, y*4), 1, (tile_size, tile_size))
                lbl_tile_np = np.array(lbl_tile.split()[0]) # Only consider the R channel
                if np.count_nonzero(lbl_tile_np) < 157286: # 15% of 1024x1024
                    continue
                tile = slide.read_region((x*4, y*4), 1, (tile_size, tile_size))
            except OpenSlideError:
                print(f"Skipping tile ({x}, {y}) for {id} due to OpenSlide error")
                continue

            __save_tile(
                tile,
                lbl_tile_np,
                save_path,
                '5x',
                f'{id}_row{y}_col{x}'
            )

    # 10x
    for y in range(0, label.level_dimensions[0][1], tile_size*2):
        for x in range(0, label.level_dimensions[0][0], tile_size*2):
            try:
                lbl_tile = label.read_region((x, y), 0, (tile_size*2, tile_size*2))
                lbl_tile_np = np.array(lbl_tile.split()[0]) # Only consider the R channel
                if np.count_nonzero(lbl_tile_np) < 629145: # 15% of 2048x2048
                    continue
                tile = slide.read_region((x, y), 0, (tile_size*2, tile_size*2))
                tile = tile.resize((tile_size, tile_size), resample=Image.NEAREST)
                lbl_tile_np = cv2.resize(lbl_tile_np, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            except OpenSlideError:
                print(f"Skipping tile ({x}, {y}) for {id} due to OpenSlide error")
                continue
    
            __save_tile(
                tile,
                lbl_tile_np,
                save_path,
                '10x',
                f'{id}_row{y}_col{x}'
            )

    # 20x
    for y in range(0, label.level_dimensions[0][1], tile_size):
        for x in range(0, label.level_dimensions[0][0], tile_size):
            try:
                lbl_tile = label.read_region((x, y), 0, (tile_size, tile_size))
                lbl_tile_np = np.array(lbl_tile.split()[0]) # Only consider the R channel
                if np.count_nonzero(lbl_tile_np) < 157286: # 15% of 1024x1024
                    continue
                tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            except OpenSlideError:
                print(f"Skipping tile ({x}, {y}) for {id} due to OpenSlide error")
                continue

            __save_tile(
                tile,
                lbl_tile_np,
                save_path,
                '20x',
                f'{id}_row{y}_col{x}'
            )


def __save_tile(
        slide_tile: Image,
        lbl_tile: np.array,
        save_path: Path,
        resolution: str,
        name: str
    ) -> None:
    """
    Save the slide tile and label tile to the provided save path with the provided name.

    Parameters
    ----------
    - `slide_tile: Image` - The slide tile to save.
    - `lbl_tile: np.array` - The label tile to save.
    - `save_path: Path` - The top level directory to save the tiles in.
    - `resolution: str` - The resolution of the tile used to determine save folder.
    - `name: str` - The name of the tile.

    Returns
    ----------
    - `None`
    """
    slide_tile.convert('RGB').save(save_path / ('img_' + resolution) / (name + '.png'))
    Image.fromarray(lbl_tile).save(save_path / ('lbl_' + resolution) / (name + '.png'))


def __add_bottom_margin(
    pil_img: Image,
    bottom: int, 
    color: tuple[int, int, int] | int
) -> Image:
    width, height = pil_img.size
    new_width = width
    new_height = height + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (0, 0))

    return result
