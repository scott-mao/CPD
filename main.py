import os
import time, datetime
import utils.common as utils
from utils.parser import get_args

import torch
import torch.backends.cudnn as cudnn

from modules.finetuner import Finetuner


args = get_args()

args.print_freq = (256*50)//args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device_ids = list(map(int, args.gpu.split(',')))
args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

torch.manual_seed(args.seed)  ##for cpu
if args.gpu:
    torch.cuda.manual_seed(args.seed)  ##for gpu

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

if len(args.device_ids) == 1:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    args.name_base='module.'
else:
    args.name_base=''

def main():
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    # model = Net(args) #create and load pretrained network

    tuner = Finetuner(args, logger) #import the network in a predefined class

    if args.cluster: #Clustering
        cluster_idx = tuner.cluster() #rank값을 아예 0으로 설정해두면 되지 않을까 (여기서 rank_w를 불러온 다음에 rank_w2를 불러오고)

    if args.prune: #Pruning
        prune_idx = tuner.prune() #rank_w2를 불러서 pruning을 함

    if args.decompose: #Filter Decomposition
        tuner.decompose()

    end_t = time.time()

if __name__ == '__main__':
  main()
