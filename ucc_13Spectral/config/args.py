import argparse

def parse_func():
    parse = argparse.ArgumentParser(
        description="Training Options")
    
    parse.add_argument(
        "--project_name",
        type=str,
        default="AI AWB for dominant color scene")

    parse.add_argument(
        "--version",
        type=str,
        default="0.0.1",
        help="the version of current dataset and model \
              should be formated as 'version.release.modifications'"
                       )

    parse.add_argument(
        "--model_name",
        type=str,
        default="ucc13",
        help="the name of the model")
    
    parse.add_argument(
        "--use_spec",
        type=int,
        default=1,
        help='whether use spectral information or not, 1 means True, others means False'
    )

    parse.add_argument(
        "--input_mode",
        type=int,
        default=0,
        help='0: follows the mode, 1: only ASL, 2: only uv-histogram'
    )
    
    parse.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or test mode"
    )

    parse.add_argument(
        "--fold_num",
        type=int,
        default=0,
        help="choose from 0, 1 ,2"
    )

    parse.add_argument(
        '--data_dir',
        type=str,
        default=r"",
        help="directory storing the training data")

    parse.add_argument(
        '--val_dir',
        type=str,
        default=r"",
        help="directory storing the validation data")

    parse.add_argument(
        '--test_dir',
        type=str,
        default="./input",
        help="directory storing the test data")

    parse.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="output directory")

    parse.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of total training epochs")

    parse.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to checkpoint")

    parse.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=
        "batch size (in distributed mode, it represents data fed to each GPU)")

    parse.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="the name of optimizer, currently support ['adam', 'adamw']")

    parse.add_argument(
        "--lr", 
        type=float, 
        default=2e-3, 
        help="learning rate")

    parse.add_argument(
        "--input_size",
        type=int,
        default=-1,
        help="input image size, -1 means without resize")

    parse.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help="the random seed in training")

    parse.add_argument(
        "--resume",
        type=int,
        default=0,
        help="whether to resume training")

    parse.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints")

    parse.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="number of workers in dataloader")

    parse.add_argument(
        "--save_iter",
        type=int,
        default=100,
        help="save model checkpoint after certain epochs")

    parse.add_argument(
        "--shuffle",
        type=int,
        default=0,
        help=
        "whether to shuffle data, results should be different only when cross validation is used"
    )

    parse.add_argument(
        "--post_process",
        type=int,
        default=1,
        help="whether to post process the predicted white point")

    parse.add_argument(
        "--tag", 
        type=str, 
        default="", 
        help="tag for the model")

    opt = parse.parse_args()
    batch_ratio = opt.batch_size / 32
    opt.lr = opt.lr * batch_ratio
    if opt.tag:
        opt.model_name = "{}#{}".format(opt.model_name, opt.tag)
    verify_args(opt)
    return opt


def verify_args(opt):
    pass
