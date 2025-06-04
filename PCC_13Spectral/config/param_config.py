import argparse

def parse_args():
    """ _arguments parsing_

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, help='Number of epochs', default=1000)
    parser.add_argument('-batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('-lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('-fold_num', type=int, help='Use three cross-validation. '
                                                    'The number should be changed manually'
                                                    ' from : 0, 1, 2', default=0)
    parser.add_argument('-input_mode',type=int,default=0,help='0:follow the mode, 1: only ASL, 2: only semantic')
    parser.add_argument('-data_name', type=str, help='datasets', default='O1_Pure_New')
    parser.add_argument('-seed', type=int, help='Default seed', default=666)
    parser.add_argument('-num_workers', type=int, help='Influence the speed of dataset loader', default=4)
    parser.add_argument('-data_path', type=str, help='dataset path', default='./dataset/O1_Pure_New')
    parser.add_argument('-mode', type=str, help='chose in ["semantic","spectral","semantic_spectral"]', default='semantic_spectral')
    parser.add_argument('--output', action='store_true', default=True, help="shows output")

    args = parser.parse_args()

    if args.output:
        print(f'dataset path:   {args.data_path}')
        print(f'num_workers:    {args.num_workers}')
        print(f'batch_size:     {args.batch_size}')
        print(f'epochs :        {args.epochs}')
        print(f'learning rate : {args.lr}')
        print(f'manual_seed:    {args.seed}')
        print(f'fold number:    {args.fold_num}')
        print(f'mode:           {args.mode}')

    return args
