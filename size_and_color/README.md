# Does Size (Resolution) and Color (Channels) Matter in (Computational) Pathology
This project was used to provide a first estimate if color (channels) and (high) resolution are necessary/beneficial for computational pathology.

## Datasets Used
For the question `Does Size Matter in Pathology` the [PANDA](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data?select=train_images) dataset was used.

For the question `Does Color Matter in Pathology` the [PANDA](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data?select=train_images), [RINGS](https://data.mendeley.com/datasets/h8bdwrtnr5/1), and [CoCaHis](https://cocahis.irb.hr) datasets were used.

## Reproduce the Experiment/Results
If you only want to run the pre-trained models step *41.* can be omitted.<br>
If you only want to reproduce the results (run the final evaluation) no download or training is required. You only have to run step *2.* and *5.*.

The project was created and tested with Python 3.12. When using other Python versions the behaviour of the code may differ.

The datasets used for this project require XX GB of storage. If your disk does not have enough storage the data can also be downloaded to any
other disk/external drive. The structure of the `data` must be kept the same to allow the automatic processing.

1. Download the datasets and move them to the respective folder in the `size_and_color/data/raw` folder (approximately 214GB overall). 
It should look like this:
    ```
    data/raw
    |-> cocahis
    | '-> CoCaHis.hdf5
    |-> panda
    | '-> prostate-cancer-grade-assessment.zip
    '-> rings
      '-> RINGS algorithm dataset.zip
    ```
2. Install the requirements for this project using `pip install -r requirements.txt`
    - the usage of a Python virtual environment is advised
3. Run the data setup script `python setup_data.py` 
    - if the download location was changed provide the path using the `-dp`/`--data-path` flag
    - e.g., `python setup_data.py -dp /Volumes/external_drive/data/`
    - the extracted data will require approximately GB of storage
    - when run with the `-rr` flag the raw data will be removed during setup
4. Train/Test the models
    
    41. Train the models yourself on the training data using the training script `python train_model.py` 
        - using `-h` \ `--help` will provide you with further information
    42. Evaluate your/the pre-trained models on the test data using the evaluate model script `python evaluate_model.py`
        - using `-h` \ `--help` will provide you with further information

5. Answer the research question (overall evaluation and comparison of models)
    - EITER: run the *Size and Color, Does it Matter in Pathology* script `python sac_dimip.py`
        - using `-h` \ `--help` will provide you with further information
        - the script will provide some evaluations and generate `csv` and `png` files highlighting evaluation results
    - ALTERNATIVELY: run the *Size and Color, Does it Matter in Pathology* notebook `python -m notebook sac_dimip.ipynb`
        - this provides a more interactive and GUI-driven approach to the overall evaluation
