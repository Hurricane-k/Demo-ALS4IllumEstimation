
#%% import python package
import logging
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from scipy.io import savemat
import argparse

from src.c5 import network
from src.datasetC5 import DataC5

#%%
def test_net(
        net,
        device,
        dir_img,
        fold,
        multiple_test=False,
        data_num=7,
        batch_size=1,
        model_name='C5_model',
        input_size=64,
        load_hist=False,
        use_spec=False,
        use_sepc_input=0, 
        save_output = True,
        dir_saveoutput = '',
        g = True):
    """ copy and paste from the inital C5 github

        parameters:
        use_spec: whether to use Spectral_info or not
            True: use the ASL 13-channel info
            False: not use ASL 13-channel info
        use_sepc_input:
            0: following model
            1: only using ASL
            2: only using uv-histogram
        dir_saveoutput: folder to save results
    """
    
    test = DataC5(dir_img = dir_img, 
                fold_num = fold,
                input_size=input_size, 
                mode='testing',
                data_num=data_num,
                load_hist=load_hist)

    test_loader = DataLoader(test, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=8, 
                            pin_memory=True) 

    logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Test Img File:         {dir_img}
        Batch size:            {batch_size}
        use Spec or not:       {use_spec}
        Number of input:       {data_num}
        Learn G multiplier:    {g}
        Input size:            {input_size} x {input_size}
        Device:                {device.type}
    ''') 

    if multiple_test:
        number_of_tests = 10
    else:
        number_of_tests = 1

    with torch.no_grad():
        for test_i in range(number_of_tests):
            results = np.zeros((len(test), 3))  # to store estimated illuminant values
            gt = np.zeros((len(test), 3))  # to store ground-truth illuminant colors
            filenames = []  # to store filenames
            index = 0

            for batch in test_loader:
                model_histogram = batch['model_input_histograms']
                model_histogram = model_histogram.to(device=device,
                                                    dtype=torch.float32)
                file_names = batch['file_name']

                histogram = batch['histogram']
                histogram = histogram.to(device=device, 
                                        dtype=torch.float32)

                gt_ill = batch['gt_ill']
                gt_ill = gt_ill.to(device=device, 
                                dtype=torch.float32)
                

                if use_spec:
                    spec_info = batch['spec_info']
                    spec_info = spec_info.to(device=device,
                                            dtype=torch.float32)
                    
                    if use_sepc_input==1:
                        # print('input: {}'.format(file_names))
                        # print('only use ASL')
                        # print('---------')
                        histogram = torch.zeros_like(histogram)
                        histogram = histogram.to(device=device, 
                                                 dtype=torch.float32)
                    elif use_sepc_input==2:
                        # print('input: {}'.format(file_names))
                        # print('only use uv-histogram')
                        # print('---------')
                        spec_info = torch.zeros_like(spec_info)
                        spec_info = spec_info.to(device=device, 
                                                 dtype=torch.float32)
                    

                    predicted_ill, _, _, _, _ = net(histogram, 
                                                    model_in_N=model_histogram,
                                                    ASL_info = spec_info)

                else:
                    predicted_ill, _, _, _, _ = net(histogram, 
                                                    model_in_N=model_histogram)

                L = len(predicted_ill)
                results[index:index + L, :] = predicted_ill.cpu().numpy()
                gt[index:index + L, :] = gt_ill.cpu().numpy()
                for f in file_names:
                    filenames.append(f)
                index = index + L

            if save_output:
                if use_sepc_input == 0:
                    save_dir = os.path.join('results', model_name)
                elif use_sepc_input in [1,2] and dir_saveoutput != '':
                    save_dir = os.path.join('results', dir_saveoutput)
                else:
                    save_dir = os.path.join('results', model_name)
                if not os.path.exists(save_dir):
                    if not os.path.exists('results'):
                        os.mkdir('results')
                    os.mkdir(save_dir)
                    logging.info(f'Created results directory {save_dir}')
                if multiple_test:
                    savemat(os.path.join(save_dir, f'gt_{test_i + 1}.mat'), 
                            {'gt': gt})
                    savemat(os.path.join(save_dir, f'results_{test_i + 1}.mat'),
                            {'predicted': results})
                    savemat(os.path.join(save_dir, f'filenames_{test_i + 1}.mat'),
                            {'filenames': filenames})
                else:
                    savemat(os.path.join(save_dir, 'gt.mat'), {'gt': gt})
                    savemat(os.path.join(save_dir, 'results.mat'), {'predicted': results})
                    savemat(os.path.join(save_dir, 'filenames.mat'), {'filenames': filenames})

    logging.info('End of testing')

#%%
def get_args():
  
    parser = argparse.ArgumentParser(description='Test C5.')

    parser.add_argument('-b', '--batch-size', 
                        metavar='B', type=int,
                        nargs='?', default=1,
                        help='Batch size', 
                        dest='batchsize')

    parser.add_argument('-s', '--input-size', 
                        dest='input_size', type=int,
                        default=64, 
                        help='Size of input (hist and image)')

    parser.add_argument('-us', '--use-spectral', 
                        dest='usespectral', type=int, 
                        default=0,
                        help='whether to use ASL 13-channel info or not')
    
    parser.add_argument('-usi', '--use-spectral-input', 
                        dest='usespectral_input', type=int, 
                        default=0,
                        help='0: following model, 1: using ASL, 2: using uv-histogram')

    parser.add_argument('-ntrd', '--testing-dir-in', 
                        dest='in_tedir',
                        default='./dataset/O1_Pure',
                        help='Input testing image directory')

    parser.add_argument('-lh', '--load-hist', 
                        dest='load_hist',
                        type=bool, default=True,
                        help='Load histogram if exists')

    parser.add_argument('-dn', '--data-num', 
                        dest='data_num', type=int, default=7,
                        help='Number of input data for calibration')

    parser.add_argument('-lg', '--g-multiplier', 
                        type=bool, default=False,
                        help='Have a G multiplier', 
                        dest='g_multiplier')

    parser.add_argument('-mt', '--multiple_test', 
                        type=bool, default=False,
                        help='do 10 tests and save the results',
                        dest='multiple_test')

    parser.add_argument('-n', '--model-name', 
                        dest='model_name',
                        default='c5_model')

    parser.add_argument('-g', '--gpu', 
                        dest='gpu', 
                        default=0, 
                        type=int)

    return parser.parse_args()

#%%
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Testing C5')

    args = get_args()

    device = torch.device('cuda' 
                          if torch.cuda.is_available() 
                          else 'cpu')
    
    if device.type != 'cpu':
        torch.cuda.set_device(args.gpu)

    logging.info(f'Using device {device}')

    if args.usespectral:
        usespectral = True
        model_name = 'C5_wSpec'
    else:
        usespectral = False
        model_name = 'C5_woSpec'

    net = network(
        input_size=args.input_size, 
        learn_g=args.g_multiplier,
        data_num=args.data_num, 
        device=device)
    
    dataset_name = os.path.basename(args.in_tedir)

    for fold in range(3):
        model_path = os.path.join('./models', 
                                model_name + '_' + dataset_name +
                                f'_fold_{fold + 1}.pth')
        # updated by 2024.09.11 add new parameter 'strict=False';
        net.load_state_dict(torch.load(model_path, 
                                    map_location=device,
                                    weights_only=False))
        logging.info(f'Model loaded from {model_path}')
        net.to(device=device)
        net.eval()

        if args.usespectral_input not in [1,2]:
            dir_saveoutput = (model_name + 
                              '_' + 
                              dataset_name + 
                              f'_fold_{fold + 1}')
        elif args.usespectral_input == 1:
            dir_saveoutput = (model_name + 
                              '_' + 
                              'input_only_ASL' + 
                              '_' + 
                              dataset_name + 
                              f'_fold_{fold + 1}')
        elif args.usespectral_input == 2:
            dir_saveoutput = (model_name + 
                              '_' + 
                              'input_only_hist' + 
                              '_' + dataset_name +
                              f'_fold_{fold + 1}')

        test_net(
            net,
            device,
            dir_img=args.in_tedir,
            fold = fold,
            multiple_test=args.multiple_test,
            data_num=args.data_num,
            batch_size=args.batchsize,
            model_name=model_name + '_' + dataset_name +f'_fold_{fold + 1}',
            input_size=args.input_size,
            load_hist=args.load_hist,
            use_spec=usespectral,
            use_sepc_input=args.usespectral_input,
            save_output = True,
            dir_saveoutput = dir_saveoutput,
            g = args.g_multiplier)