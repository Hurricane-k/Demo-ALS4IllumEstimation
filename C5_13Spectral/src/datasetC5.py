import torch
import logging
from glob import glob
import os
import numpy as np
import math
import random

from src import ops




class DataC5(torch.utils.data.Dataset):

    def __init__(self,
                 dir_img = './dataset/O1_Pure',
                 cv_or_not = True,
                 fold_num = 0,
                 data_num = 7,
                 mode = 'training',
                 input_size=64,
                 load_hist=True,
                 image_size = [256, 384]):
        """_summary_

        Args:
            dir_img (str, optional): _the root dir of the dataset_.
            cv_or_not (bool, optional): _cross validation or not_.
            fold_num (int, optional): _valid for cross validation_.
            data_num (int, optional): _the number of images for one input_.
            mode (str, optional): _affect the loaded images_.
            input_size (int, optional): _the size of histogram_.
            load_hist (bool, optional): _if need to reload histogram of pic_.
            image_size (list, optional): _the size of input image_.
        """
        self.dir_img = dir_img
        self.cv_or_not = cv_or_not
        self.fold_num = fold_num
        self.data_num = data_num
        self.mode = mode
        self.input_size=input_size
        self.additional_data_num = data_num - 1
        self.load_hist=load_hist
        self.image_size = image_size
        self.from_rgb = ops.rgb_to_uv  # rgb to chroma conversion function
        self.to_rgb = ops.uv_to_rgb  # chroma to rgb conversion function
        self.hist_boundary = ops.get_hist_boundary()

        # cross-validation 3 folds
        if not self.cv_or_not:
            self.imgfiles = glob(os.path.join(self.dir_img,
                                        'numpy_data',
                                        '*.npz'))
        else:
            data,_ = DataC5.cv_files(self.dir_img)
            if self.fold_num == 0 and self.mode == 'training':
                self.imgfiles = data[1]+data[2]
            elif self.fold_num == 0 and self.mode == 'testing':
                self.imgfiles = data[0]
            elif self.fold_num == 1 and self.mode == 'training':
                self.imgfiles = data[0]+data[2]
            elif self.fold_num == 1 and self.mode == 'testing':
                self.imgfiles = data[1]
            elif self.fold_num == 2 and self.mode == 'training':
                self.imgfiles = data[0]+data[1]
            else:
                self.imgfiles = data[2]

        assert (self.data_num >= 1)
        assert (self.mode == 'training' or self.mode == 'testing')
        assert (self.input_size % 2 == 0)

    def __len__(self):
        """ Gets length of image files in the dataloader. """
        return len(self.imgfiles)
    
    def __getitem__(self, i):
        """_summary_

        Args:
            i (_int_): _the index of the input image_

        Raises:
            Exception: addictive images are not enough due to the lack of images_

        Returns:
            _dict_: _saved all info_

            the important input data is additional_histogram
            the shape is (num_data, channel, height, width)
            num_data is the number of images (1 target + n additive)
            channel: the number of histogram
                (one for inital image)
                (one for edge info of the inital image)
                (one for u_coord, same for all images)
                (one for v_coord, same for all images)
        """

        img_file = self.imgfiles[i]

        in_img = np.load(img_file)['pureStatsMap']
        spc_img = np.load(img_file)['spectral13']
        spc_img = np.reshape(spc_img/spc_img.sum(),(-1,1,1))

        in_img = ops.resize_image(in_img, 
                                  self.image_size)
        
        # useless, the following command line
        rgb_img = ops.to_tensor(in_img)  # for visualization

        # load the groundtruth of white point
        ill_file =  img_file.replace('numpy_data','numpy_labels')
        ill_file =  ill_file.replace('npz','npy')
        gt_ill = np.load(ill_file)
        gt_ill = gt_ill/np.linalg.norm(gt_ill)
        gt_ill = torch.from_numpy(gt_ill)

        # computes histogram feature of rgb and edge images
        # flexible change the size of histogram
        if self.input_size == 64:
            post_fix = ''
        else:
            post_fix = f'_{self.input_size}'

        # calculate the histogram of the current image
        if os.path.exists(os.path.splitext(img_file)[0] +
                          f'_histogram{post_fix}.npy') and self.load_hist:
            histogram = np.load(os.path.splitext(img_file)[0] +
                                f'_histogram{post_fix}.npy', 
                                allow_pickle=False)
        else:
            histogram = np.zeros((self.input_size, 
                                  self.input_size, 2))
            valid_chroma_rgb, valid_colors_rgb = ops.get_hist_colors(
                in_img, self.from_rgb)
            histogram[:, :, 0] = ops.compute_histogram(
                valid_chroma_rgb, 
                self.hist_boundary, 
                self.input_size,
                rgb_input=valid_colors_rgb)

            edge_img = ops.compute_edges(in_img)
            valid_chroma_edges, valid_colors_edges = ops.get_hist_colors(
                edge_img, self.from_rgb)

            histogram[:, :, 1] = ops.compute_histogram(
                valid_chroma_edges, 
                self.hist_boundary, 
                self.input_size,
                rgb_input=valid_colors_edges)

            np.save(os.path.splitext(img_file)[0] + 
                    f'_histogram{post_fix}.npy',
                    histogram)
            
        # the current histogram
        in_histogram = ops.to_tensor(histogram)

        # gets additional input data
        if self.additional_data_num > 0:
            additiona_files = DataC5.get_rand_examples_from_sensor(
            current_file=img_file, 
            files=self.imgfiles,
            target_number=self.additional_data_num)
        else:
            additiona_files = None

        # histogram fromt the current image
        additional_histogram = histogram

        u_coord, v_coord = ops.get_uv_coord(self.input_size,
                                            tensor=False, 
                                            normalize=True)
        # 4 channel (img, edge_img, u_coord, v_coord)
        # I think the last two channels are unnecessary
        # because they are same for every input
        u_coord = np.expand_dims(u_coord, axis=-1)
        v_coord = np.expand_dims(v_coord, axis=-1)

        additional_histogram = np.concatenate([additional_histogram, 
                                            u_coord],
                                            axis=-1)
        additional_histogram = np.concatenate([additional_histogram, 
                                            v_coord],
                                            axis=-1)
        additional_histogram = np.expand_dims(additional_histogram, 
                                            axis=-1)
        
        # if multiple input is used, load them
        # the loop copy and paste from the current image
        if additiona_files is not None:
            for file, i in zip(additiona_files, 
                               range(len(additiona_files))):
                # computes histogram feature of rgb and edge images
                if os.path.exists(os.path.splitext(file)[0] +
                                  f'_histogram{post_fix}.npy') and self.load_hist:
                    histogram = np.load(os.path.splitext(file)[0] +
                                        f'_histogram{post_fix}.npy', 
                                        allow_pickle=False)

                else:
                    img = np.load(img_file)['pureStatsMap']
                    h, w, _ = img.shape
                    if h != self.image_size[1] or w != self.image_size[0]:
                        img = ops.resize_image(img, 
                                               self.image_size)
                    histogram = np.zeros((self.input_size, 
                                          self.input_size, 2))
                    valid_chroma_rgb, valid_colors_rgb = ops.get_hist_colors(
                        img, 
                        self.from_rgb)
                    histogram[:, :, 0] = ops.compute_histogram(
                        valid_chroma_rgb, 
                        self.hist_boundary, 
                        self.input_size,
                        rgb_input=valid_colors_rgb)
                    edge_img = ops.compute_edges(img)
                    valid_chroma_edges, valid_colors_edges = ops.get_hist_colors(
                        edge_img,self.from_rgb)

                    histogram[:, :, 1] = ops.compute_histogram(
                        valid_chroma_edges, 
                        self.hist_boundary, self.input_size,
                        rgb_input=valid_colors_edges)

                    np.save(os.path.splitext(file)[0] + f'_histogram{post_fix}.npy',
                            histogram)

                histogram = np.concatenate([histogram, u_coord], axis=-1)
                histogram = np.concatenate([histogram, v_coord], axis=-1)
                histogram = np.expand_dims(histogram, axis=-1)

                additional_histogram = np.concatenate([additional_histogram, histogram],
                                                        axis=-1)

        additional_histogram = ops.to_tensor(additional_histogram, dims=4)

        return {'image_rgb': rgb_img,
                'file_name': os.path.basename(img_file),
                'histogram': in_histogram,
                'model_input_histograms': additional_histogram,
                'spec_info': spc_img,
                'gt_ill': gt_ill}
    
    @staticmethod
    def get_rand_examples_from_sensor(current_file, 
                                      files, 
                                      target_number):
        """_inherited and modified from C5_

        Args:
            current_file (_str_): _the chosen img files_
            files (_type_): _all files located at the same folder_
            target_number (_type_): _the number of additive images_

        Raises:
            Exception: _not enough files as additive images_

        Returns:
            list: _containing additive images_
        """

        sensor_name = current_file.split('/')[-3]
        sensor_files = [file for file in files if sensor_name in file]
        sensor_files.remove(current_file)
        random.shuffle(sensor_files)
        if len(sensor_files) < target_number:
            raise Exception('Cannot find enough training data from sensor:'
                            f'{sensor_name}')
        return sensor_files[:target_number]

    @staticmethod
    def cv_files(dir_img):
        """_cross validation for 3 folds_

        Args:
            dir_img (_str_): _the root dict of the DATASET_

        Returns:
            data(_list_): _a list with 3 lists_
        """

        if not os.path.exists('folds'):
            os.mkdir('folds')
            logging.info('Created cross validation folds directory')

        dataset_name = os.path.basename(dir_img)
        if (os.path.exists(f'folds/{dataset_name}_fold_1.npy') and
            os.path.exists(f'folds/{dataset_name}_fold_2.npy') and
            os.path.exists(f'folds/{dataset_name}_fold_3.npy')):
            logging.info('Loading CV folds...')
            testing_fold_1_filenames = np.load(f'folds/{dataset_name}_fold_1.npy')
            testing_fold_2_filenames = np.load(f'folds/{dataset_name}_fold_2.npy')
            testing_fold_3_filenames = np.load(f'folds/{dataset_name}_fold_3.npy')

            testing_fold_1 = [os.path.join(dir_img,'numpy_data', 
                                            os.path.basename(file))
                                            for file in testing_fold_1_filenames]
            testing_fold_2 = [os.path.join(dir_img,'numpy_data', 
                                            os.path.basename(file))
                                            for file in testing_fold_2_filenames]
            testing_fold_3 = [os.path.join(dir_img,'numpy_data', 
                                            os.path.basename(file))
                                            for file in testing_fold_3_filenames]

        # if cv files are not exist, create new cv indices; save them in 'folds'
        # directory.
        else:
            input_files = glob(os.path.join(dir_img,
                                            'numpy_data',
                                            '*.npz'))
            random.shuffle(input_files)
            testing_fold_1 = input_files[:math.ceil(len(input_files) * 1 / 3)]
            testing_fold_2 = input_files[math.ceil(len(input_files) * 1 / 3):
                                        math.ceil(len(input_files) * 2 / 3)]
            testing_fold_3 = input_files[math.ceil(len(input_files) * 2 / 3):]
            np.save(f'folds/{dataset_name}_fold_1.npy', testing_fold_1)
            np.save(f'folds/{dataset_name}_fold_2.npy', testing_fold_2)
            np.save(f'folds/{dataset_name}_fold_3.npy', testing_fold_3)

        data = [testing_fold_1, testing_fold_2, testing_fold_3]
        folds = 3

        return data, folds
