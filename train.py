import argparse
from train_model import Train
from utils import *


parses = argparse.ArgumentParser()
parses.add_argument('--config', type=str, default='configs/train.yaml', help='Path to the configs file.')
opts = parses.parse_args()

def train():
    args = get_config(opts.config)
    if args is None:
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args['GPU_ID']))
    model = Train(args)
    model.train()
    pass

if __name__ == '__main__':
    train()
