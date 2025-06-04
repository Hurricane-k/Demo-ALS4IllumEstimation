#%% python package import
import torch
import os
import json
import logging
from collections import OrderedDict
from src.ucc13 import UCC
from utils.uccDataset import uccDataset
from utils.metrics import angular_error_torch, metrics_torch, NumpyEncoder
from torch import optim, utils
from config.settings import DEVICE, reproduce
from config.args import parse_func


#%%
if __name__ == '__main__':

    # load necessary path
    torch.set_float32_matmul_precision('high')
    args = parse_func()
    model_name = args.model_name
    if args.use_spec == 1:
        use_spec = True
    else:
        use_spec = False
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    num_workers = args.num_workers

    dataset_name = (data_dir.split('/')[-1] 
                    if data_dir.split('/')[-1] != '' 
                    else data_dir.split('/')[-2])

    if use_spec:
        project_name = dataset_name+'_'+model_name+'_'+'wSpectral'
    else:
        project_name = dataset_name+'_'+model_name+'_'+'woSpectral'

    if use_spec == False:
        input_mode = 0
    else:
        if args.input_mode not in [0,1,2]:
            input_mode = 0
        else:
            input_mode = args.input_mode
    
    if input_mode == 0:
        save_project_name = project_name
    elif input_mode == 1:
        save_project_name = project_name + '_input_only_ASL'
    else: # input_mode == 2
        save_project_name = project_name + '_input_only_semantic'


    logging.info(f'''
        Project Name:     {project_name}
        Model Name  :     {model_name}
        Use Sepctral:     {use_spec}
        DataSet     :     {data_dir}
        Save Dictionary:  {save_project_name}
    ''')

    for fold_num in range(3):
        # locate the path of model
        fold = fold_num
        model_path = os.path.join(
            checkpoint_path,
            project_name,
            'fold_'+str(fold),
            'best.ckpt'
        )

        # load model
        assert os.path.exists(model_path), 'check the path of models'
        ucc13_test = UCC()
        model_para = torch.load(model_path,weights_only=True)['state_dict']
        def modify_key(key):
            return key.replace('model.','',1)
        model_para_modified = OrderedDict((modify_key(k), v) 
                                        for k, v 
                                        in model_para.items())
        ucc13_test.load_state_dict(model_para_modified)

        # check if the trained_parameters are loaded successfully
        for name_weight, weight in ucc13_test.named_parameters():
            if torch.equal(model_para_modified[name_weight],
                        weight.to(DEVICE)):
                pass
            else:
                raise TypeError('the trained model mismatch the model structure') 
                
        # load dataset
        test_dataset = uccDataset(
            data_dir,
            mode='test', 
            transform=None,
            fold = fold)

        val_loader = utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False)

        # save results
        dict_result = {}
        torch_ae = []
        ucc13_test.eval()
        with torch.no_grad():
            for data in val_loader:
                illum = data['illum']
                input_x1 = data['input']
                input_x2 = data['spect']
                if use_spec:
                    if input_mode == 1: # ASL
                        input_x1 = torch.zeros_like(input_x1)
                    elif input_mode == 2:
                        input_x2 = torch.zeros_like(input_x2)
                    else:
                        pass
                    estim = ucc13_test.inference(input_x1,input_x2)
                else:
                    estim = ucc13_test.inference(input_x1,None)
                ae = angular_error_torch(estim, illum)
                torch_ae.append(ae)
                dict_result[data['name'][0]] = {
                    'gt': (illum/torch.norm(illum,dim=-1)).numpy().squeeze(),
                    'est': (estim/torch.norm(estim,dim=-1)).numpy().squeeze(),
                    'ae': ae.numpy()
                }

        dict_metrics = metrics_torch(torch.Tensor(torch_ae))
        dict_result['metrics'] = dict_metrics


        logging.info(f'''
            fold       {fold}
            mean       {dict_metrics['mean']:.4f}
            median     {dict_metrics['median']:.4f}
            trimean    {dict_metrics['trimean']:.4f}
            bst.25%    {dict_metrics['bst25']:.4f}
            wst.25%    {dict_metrics['wst25']:.4f}
            wst.05%    {dict_metrics['wst05']:.4f}
                    ''')

        result_path_fold = os.path.join(
            save_path,
            save_project_name,
            'fold_'+str(fold),
        )

        os.makedirs(result_path_fold, exist_ok=True)

        with open(os.path.join(result_path_fold,'result_val.json'),
                'w') as f:
            json.dump(dict_result,f,indent=4, cls=NumpyEncoder)
        del f

