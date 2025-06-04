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