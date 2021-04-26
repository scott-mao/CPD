import os
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='total number of epoch for fine-tuning after one-shot pruning.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size.')

    parser.add_argument(
        '--lr',
        default=0.01,
        type=float,
        help='initial learning rate')

    parser.add_argument(
        '--lr_decay_step',
        default='20, 40, 60, 80, 100, 120, 140, 160, 180',
        type=str,
        help='learning rate decay step')

    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='momentum')

    parser.add_argument(
        '--wd',
        default=5e-4,
        type=float,
        help='weight decay')

    parser.add_argument(
        '--job_dir',
        type=str,
        default='./result/resnet_56/test1',
        help='load the model from the specified checkpoint')

    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether continue training from the same directory')

    parser.add_argument(
        '--use_pretrain',
        # action='store_true',
        default=True,
        help='whether use pretrain model')

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        default='./checkpoints/',
        help='pretrain model path')

    parser.add_argument(
        '--gpu',
        type=str,
        default='2, 3, 4, 5',
        help='Select gpu to use')

    parser.add_argument(
        '--compress_rate',
        type=str,
        default=None,  # default pruning ratio
        # default='[0.95]+[0.5]*6+[0.9]*4+[0.8]*2', #vgg_16_bn
        # default='[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0]', #densenet_40
        # default='[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]', #resnet56
        # default='[0.1]+[0.40]*36+[0.40]*36+[0.4]*36',  # resnet110
        # default = '[0.10]+[0.8]*5+[0.85]+[0.8]*3', #googlenet
        help='compress rate of each conv')

    parser.add_argument(
        '--rank_conv_prefix',
        type=str,
        default='./rank_conv/',
        help='rank conv file folder')

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        # default='imagenet',
        choices=('cifar10', 'imagenet'),
        help='dataset')


    parser.add_argument(
        '--resume_mask',
        type=str,
        default=None,
        help='mask loading')

    parser.add_argument(
        '--start_cov',
        type=int,
        default=0,
        help='The num of conv to start prune')

    parser.add_argument(
        '--arch',
        type=str,
        default='googlenet',
        choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet', 'resnet_50', 'mobilenet_v1', 'mobilenet_v2'),
        help='The architecture to prune')

    parser.add_argument(
        '--pr_step',
        type=float,
        default=0.05,
        help='compress rate of each conv')
    parser.add_argument(
        '--total_pr',
        type=float,
        default=0.75,
        help='pruning ratio')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='The num of batch to get rank.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='num of workers for dataloader'
    )
    parser.add_argument(
        '--energy',
        type=bool,
        default=False,
        help='use energy as a criterion'
    )

    parser.add_argument(
        '--cluster',
        type=bool,
        default=False,
        help='switch for filter clustering'
    )

    parser.add_argument(
        '--prune',
        type=bool,
        default=True,
        help='switch for filter pruning'
    )

    parser.add_argument(
        '--decompose',
        type=bool,
        default=False,
        help='switch for filter decomposition'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='whether it is test mode')

    args = parser.parse_args()
    args.rank_conv_prefix = args.rank_conv_prefix + args.arch + '_limit10'
    args.pretrain_dir = args.pretrain_dir + args.arch + '.pt'

    # Data Acquisition
    args.data_dir = {
        "cifar10": './data',  # CIFAR-10
        "imagenet": '/ssd8/jhhwang/imagenet',  # ImageNet
    }[args.dataset]

    if args.arch == 'resnet_50':
        args.resume = args.resume + 'h'

    return args


