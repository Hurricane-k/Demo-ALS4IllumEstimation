import subprocess
import os
import shutil

if __name__ == "__main__":

    NameDataset = 'Dataset_Normal'
    PathDataset = os.path.join('./dataset',NameDataset)
    NameMode = 'spectral' # chosen for ['semantic_spectral','spectral','semantic']
    
    for i in range(3):
        cmd_fold = "python train.py -fold_num "+str(i)+" -data_name "+NameDataset+" -data_path "+PathDataset+" -mode "+NameMode
        print('---')
        print('')
        print(cmd_fold)
        print('')
        print('---')

        subprocess.run(cmd_fold, shell=True)

        print('===DONE!===')
        print('')

    # model for ./log to ./models
    PathFolder_src = os.path.join('./log',NameDataset)
    PathFolder_dst = os.path.join('./models',NameDataset+'_'+NameMode)
    shutil.move(PathFolder_src, PathFolder_dst)

    