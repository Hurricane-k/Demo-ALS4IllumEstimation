# necessary function in this fold

import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import json

#%% load some data from json file list
def get_data_sorted(lisScene, pathgenroot, suffix_file='_1x.json', key_json='WhitePoint', key_sub_json=None):

    """ load json file

    """

    for idx, eachScene in enumerate(lisScene):
        if pathgenroot != None:
            file = os.path.join(pathgenroot,eachScene+suffix_file)
        else:
            file = eachScene

        def extract_info_json(filejson,key_info,key_sub_json=None):
            with open(filejson,'r') as f:
                data = json.load(f)
                if key_sub_json == None:
                    data_target = data[key_info]
                else:
                    data_target = data[key_info][key_sub_json]
            del f, data
            return data_target
        
        data_fetch = np.array(extract_info_json(file,key_json,key_sub_json))

        try:
            arr_data[idx,:] = data_fetch.reshape(1,-1)
        except:
            arr_data = np.zeros((len(lisScene),data_fetch.shape[0]))
            arr_data[idx,:] = data_fetch.reshape(1,-1)
    
    return arr_data


#%% see two kinds of ALS as grey-scale image, calculate their similaiity
def ssim_similarity(matrix1, matrix2):
    # Ensure matrices are 2D and have the same shape
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"
    
    # Determine the data range based on the maximum value in the matrices
    data_range = max(matrix1.max(), matrix2.max()) - min(matrix1.min(), matrix2.min())
    
    # Compute SSIM
    similarity, _ = ssim(matrix1, matrix2, data_range=data_range, full=True)
    return similarity


#%% elbow method
def elbow_method_show(arr_data, interval_num_cluster = 5, max_n_cluster=None):
    """ elbow method for data

    arr_data (np.ndarray): np.ndarray with the shape of (N,M)
        N: the number of data
        M: the dimension of data
        e.g. one vector (r1, g1, b1, r2, g2, b2, ...), with N data

    interval_num_cluster: the common difference of an arithmetic sequence

    """
    if max_n_cluster == None:
        if arr_data.shape[0]//2 > 26:
            lis_num_cluster = list(range(5,arr_data.shape[0]//2,interval_num_cluster))
        else:
            lis_num_cluster = list(range(2,27,2))
    else:
        lis_num_cluster = list(range(10,max_n_cluster,interval_num_cluster))

    lis_distortion = []
    for num_cluster in lis_num_cluster:
        kmeans = KMeans(n_clusters=num_cluster,n_init='auto').fit(arr_data)
        lis_distortion.append(kmeans.inertia_)
        del kmeans

    plt.plot(lis_num_cluster,lis_distortion)
    plt.xlabel('num_cluster')
    plt.ylabel('distortion')
    plt.title('elbow method')
    del lis_num_cluster, lis_distortion

#%% n_cluster optimal for gap statistic
def compute_gap_statistic(data, max_k=10, B=10):
    """
    Compute the Gap Statistic for K-Means clustering.

    Parameters:
    - data: The input data as a NumPy array.
    - max_k: The maximum number of clusters to test.
    - B: The number of reference datasets to generate.

    Returns:
    - gaps: The gap values for each number of clusters.
    - optimal_k: The estimated optimal number of clusters.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Calculate the bounding box of the data
    mins = np.min(data_scaled, axis=0)
    maxs = np.max(data_scaled, axis=0)

    # Initialize variables
    gaps = np.zeros(max_k)
    s_k = np.zeros(max_k)

    for k in range(1, max_k + 1):
        # Fit KMeans and calculate within-cluster dispersion for the actual data
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        Wk = np.log(np.sum(np.min(pairwise_distances(data_scaled, kmeans.cluster_centers_, metric='euclidean')**2, axis=1)))

        # Generate B reference datasets and calculate their dispersions
        Wk_b = np.zeros(B)
        for b in range(B):
            # Generate a random reference dataset
            np.random.seed(42)
            random_data = np.random.uniform(mins, maxs, data_scaled.shape)
            kmeans.fit(random_data)
            Wk_b[b] = np.log(np.sum(np.min(pairwise_distances(random_data, kmeans.cluster_centers_, metric='euclidean')**2, axis=1)))

        # Calculate the gap statistic
        gap_k = np.mean(Wk_b) - Wk
        gaps[k - 1] = gap_k
        s_k[k - 1] = np.std(Wk_b) * np.sqrt(1 + 1/B)

    # Determine the optimal number of clusters
    optimal_k = np.argmax(gaps[:-1] >= gaps[1:] - s_k[1:]) + 1

    return gaps, optimal_k


#%%
def data_chosen_KMeans(arr_data, num_clusters=20):
    # select chosen points based on K-Means method
    """ choose data from the initial dataset based on centriors of K-Means

    Args:
        arr_data (_np.ndarray_): _(N,M) the shape_
            N: the number of data (N)
            M: the vector, the dimension of data (M)
        num_clusters (int, optional): _integer_. Defaults to 20.

    Returns:
        arr_datachosen (_np.ndarray_): the ndarray of the chosen data
        lis_idx (_list_) includes indices of the chosen data 
    """

    centroids = KMeans(n_clusters=num_clusters,n_init='auto',random_state=42).fit(arr_data).cluster_centers_
    arr_datachosen = np.zeros(centroids.shape)
    lis_idx = []

    for i in range(centroids.shape[0]):
        idxmin = np.power(arr_data-centroids[i,:],2).sum(axis=1).argmin()
        arr_datachosen[i,:] = arr_data[idxmin,:]
        lis_idx.append(idxmin)

    return arr_datachosen, lis_idx

## calculate angular error between two array
def calculate_angular_error(array1, array2):
    # Ensure the arrays have the same shape
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    
    # Normalize the vectors
    norms1 = np.linalg.norm(array1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(array2, axis=1, keepdims=True)
    
    unit_vectors1 = array1 / norms1
    unit_vectors2 = array2 / norms2
    
    # Compute the dot product for each pair of vectors
    dot_products = np.sum(unit_vectors1 * unit_vectors2, axis=1)
    
    # Clip the dot products to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Calculate the angular error in radians
    angular_errors = np.arccos(dot_products)
    
    # Optionally, convert the angular error to degrees
    angular_errors_degrees = np.degrees(angular_errors)
    
    return angular_errors, angular_errors_degrees

#%% calculate some statistic value
def cal_trimean(data):
    """_calculate tirmean_

    Args:
        data (_np.ndarray_): _all metrics_

    Returns:
        _trimean_: _trimean of input_
    """

    Q1 = np.percentile(data,25)
    median = np.percentile(data,50)
    Q3 = np.percentile(data,75)
    trimean = (Q1 + 2*median + Q3)/4

    return trimean

def cal_bst25(data):
    data_sorted = sorted(data)
    return np.mean(data_sorted[:int(len(data_sorted)*0.25)])

def cal_wst25(data):
    data_sorted = sorted(data)
    return np.mean(data_sorted[int(len(data_sorted)*0.75):])

def cal_wst05(data):
    data_sorted = sorted(data)
    return np.mean(data_sorted[int(len(data_sorted)*0.95):])