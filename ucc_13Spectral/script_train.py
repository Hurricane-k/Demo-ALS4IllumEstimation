import subprocess
import os

if __name__ == '__main__':


    NameDataset = 'Dataset_Normal'
    PathDataset = os.path.join('./dataset',NameDataset)

    for j in range(3): # fold 0, 1, 2
        cmd_sub = "python train.py --use_spec 1 --mode train --fold_num "+str(j)+" --data_dir "+PathDataset 
        print('---')
        print('')
        print(cmd_sub)
        print('')
        print('---')

        subprocess.run(cmd_sub, shell=True)

        print('===DONE===')
        print('')

