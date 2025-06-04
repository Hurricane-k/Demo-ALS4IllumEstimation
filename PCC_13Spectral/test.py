from src.utils import print_single_metric
import torch
from src.ModelOperator import ModelOperator
from src.dataset import CcData
from torch.utils.data import DataLoader
from evaluation.Evaluator import Evaluator
from config.settings import DEVICE, set_seed
from config.param_config import parse_args
import numpy as np
import os


def main(args):

    evaluator = Evaluator()
    # Model instance
    if args.mode == 'semantic_spectral':
        model = ModelOperator(in_feature=21, neurons=16)
    elif args.mode == 'spectral':
        model = ModelOperator(in_feature=13, neurons=8)
    else: # semantic
        model = ModelOperator(in_feature=8, neurons=8)

    if args.mode == 'semantic_spectral':
        mode_input = args.input_mode
    else:
        mode_input = 0

    # in case the wrong parameters passed on by args.input_mode
    if mode_input not in [0,1,2]:
        mode_input = 0

    for num_fold in range(3):
        # loading data
        fold_evaluator = Evaluator()
        data_test = CcData(args.data_path, train=False, mode=args.mode, fold_num=num_fold)
        test_loader = DataLoader(data_test, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers, drop_last=False)
        
        # parse name
        model_name = args.data_path.split('/')[2]+'_'+args.mode

        print('========Creating & Saving========')
        if mode_input == 0:
            pathresult = "./results/"+model_name+"/fold_"+str(num_fold)
        elif mode_input == 1:
            pathresult = "./results/"+model_name+"_input_only_ASL"+"/fold_"+str(num_fold)
        else: # mode_input == 2
            pathresult = "./results/"+model_name+"_input_only_semantic"+"/fold_"+str(num_fold)

        os.makedirs(pathresult, exist_ok=True)

        path_to_pretrained = "./models/"+model_name+"/" + "fold_" + str(num_fold) + "/model_cc_b1.pth"

        model.load_model(path_to_pretrained)
        model.evaluation_mode()
        print('=========Testing!========')
        arr_gt = np.zeros([len(data_test.name_data),3])
        arr_est = np.zeros([len(data_test.name_data),3])
        lis_name = []
        num_img = 0
        with torch.no_grad():
            for img, illu, name_file in test_loader:
                img, illu = img.to(DEVICE), illu.to(DEVICE)
                if mode_input == 1: # only ASL
                    img[:,:8] = 0
                elif mode_input == 2: # only semantic
                    img[:,8:] = 0
                else:
                    pass
                pred = model.predict(img)
                loss = model.get_loss(pred, illu).item()
                fold_evaluator.add_error(loss)
                evaluator.add_error(loss)
                lis_name.append(name_file)
                arr_gt[num_img,:] = np.array(illu.to('cpu'))
                arr_est[num_img,:] = np.append(np.array(pred.to('cpu')),
                                               np.array([[1-np.array(pred.to('cpu')).sum()]]),axis=1)
                num_img += 1
        np.save('/'.join([pathresult,'gt.npy']),arr_gt)
        np.save('/'.join([pathresult,'est.npy']),arr_est)
        np.save('/'.join([pathresult,'filename.npy']),np.array(lis_name))
        metrics = fold_evaluator.compute_metrics()
        print(f'---The fold_{num_fold} error---')
        print_single_metric(metrics)
    print('*****************')
    metrics = evaluator.compute_metrics()
    print('\t\t---Total error---')
    print_single_metric(metrics)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    main(args)
