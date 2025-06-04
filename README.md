<p align="center">
    <h1 align="center">
    Demo code: illuminant estimation using ALS information
    </h1>
</p>

> *ALS information*: low-resolution multispectral data of scene illuminant

## Quickstart

### Linear Transformation for Illuminant Estimation
1. `./LinearTransformation/mainFunc_Demo.ipynb` shows the framework of illuminant estimation by Linear Transformation including data split and $M$ matrix solution
2. `./necessarityFunc.py` includes two core functions including **Gap-Statstic** for hyperparameter `n_cluster` and choose training data by `K-Means`.
3. The demo dataset is Normal Dataset: `data_ALSs_NormalDataset.npy` and `data_WPs_NormalDataset.npy` stores ALS information and RGB illuminant color of corresponding scene. 

### fine-tuned C5 with different input choices
1. tested on Linux and `pytorch==2.4.1`
2. model training follows the code cell:
   ```
   cd ./C5_13Spectral
   python train.py --use-spectral 1 --learn-G True --training-dir-in ./dataset/Dataset_Normal
   ```
3. test the trained model follows the code cell:
   ```
   cd ./C5_13Spectral
   python test.py --use-spectral 1 --use-spectral-input 0 --testing-dir-in ./dataset/Dataset_Normal --g-multiplier True
   ```
4. `--use-spectral` controls input choices
   1. `--use-spectral 1` means using both ALS information and RGB images (DUAL input chocies)
   2. `--use-spectral 0` means using only RGB images (ORGB input chocies)
5. `--use-spectral-input` controls input choices, only valid for model testing
   1. `--use-spectral-input 0` follows the configuration of `--use-spectral`
   2. `--use-spectral-input 1` means test using only ALS information (OALS input choices)
   3. `--use-spectral-input 2` means test using only RGB images (ORGB)
6. the demo dataset is `Normal Dataset`: `./dataset/Dataset_Normal/numpy_data/*.npz` saves the RAW-RGB image and corresponding ALS information, and `./dataset/Dataset_Normal/numpy_labels/*.npy` saves the RGB illuminant color of scene. 
7. acknowledge [the public availibility of the initial C5 model](https://github.com/mahmoudnafifi/C5), the project `C5_13Spectral` is built upon the initial C5 model.
