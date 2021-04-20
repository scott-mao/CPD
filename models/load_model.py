import torch.nn as nn
import numpy as np
import copy
from models.cifar10.googlenet import Inception


def load_vgg_model(args, logger, model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.rank_conv_prefix+'/rank_conv_w'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight =state_dict[args.name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                rank = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[args.name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[args.name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[args.name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[args.name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_resnet_model(args, logger, model, oristate_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.rank_conv_prefix+'/rank_conv_w'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name] # parameters from the fully-connected model
                curweight =state_dict[args.name_base+conv_weight_name] #parameters from the (already) compressed model
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num: #만약에 두 필터의 사이즈가 다르면 (즉, pruning 되어져 있다면) orig > curr
                    logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[args.name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[args.name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[args.name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[args.name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[args.name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[args.name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[args.name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_google_model(args, logger, model, oristate_dict):
    state_dict = model.state_dict()

    filters = [
        [64, 128, 32, 32],
        [128, 192, 96, 64],
        [192, 208, 48, 64],
        [160, 224, 64, 64],
        [128, 256, 64, 64],
        [112, 288, 64, 64],
        [256, 320, 128, 128],
        [256, 320, 128, 128],
        [384, 384, 128, 128]
    ]

    #last_select_index = []
    all_honey_conv_name = []
    all_honey_bn_name = []
    cur_last_select_index = []

    cnt=0
    prefix = args.rank_conv_prefix+'/rank_conv_w'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, Inception):

            cnt += 1
            cov_id = cnt

            honey_filter_channel_index = [
                '.branch5x5.6',
            ]  # the index of sketch filter and channel weight
            honey_channel_index = [
                '.branch1x1.0',
                '.branch3x3.0',
                '.branch5x5.0',
                '.branch_pool.1'
            ]  # the index of sketch channel weight
            honey_filter_index = [
                '.branch3x3.3',
                '.branch5x5.3',
            ]  # the index of sketch filter weight
            honey_bn_index = [
                '.branch3x3.4',
                '.branch5x5.4',
                '.branch5x5.7',
            ]  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            last_select_index = cur_last_select_index[:]
            cur_last_select_index=[]

            for weight_index in honey_channel_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight =state_dict[args.name_base+conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = list(range(0, orifilter_num))

                for i in range(state_dict[args.name_base+conv_name].size(0)):
                    for index_j, j in enumerate(select_index):
                        state_dict[args.name_base+conv_name][i][index_j] = \
                            oristate_dict[conv_name][i][j]

                if branch_name=='_n1x1':
                    tmp_select_index = list(range(state_dict[args.name_base+conv_name].size(0)))
                    cur_last_select_index += tmp_select_index
                if branch_name=='_pool_planes':
                    tmp_select_index = list(range(state_dict[args.name_base+conv_name].size(0)))
                    tmp_select_index = [x+filters[cov_id-2][0]+filters[cov_id-2][1]+filters[cov_id-2][2] for x in tmp_select_index]
                    cur_last_select_index += tmp_select_index

            for weight_index in honey_filter_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'

                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight =state_dict[args.name_base+conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading rank from: ' + prefix + str(cov_id) + branch_name + subfix)
                    rank = np.load(prefix + str(cov_id)  + branch_name + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                else:
                    select_index = list(range(0, orifilter_num))

                for index_i, i in enumerate(select_index):
                    state_dict[args.name_base+conv_name][index_i] = \
                        oristate_dict[conv_name][i]

                if branch_name=='_n3x3':
                    tmp_select_index = [x+filters[cov_id-2][0] for x in select_index]
                    cur_last_select_index += tmp_select_index
                if branch_name=='_n5x5':
                    last_select_index=select_index

            for weight_index in honey_filter_channel_index:

                if '3x3' in weight_index:
                    branch_name='_n3x3'
                elif '5x5' in weight_index:
                    branch_name='_n5x5'
                elif '1x1' in weight_index:
                    branch_name='_n1x1'
                elif 'pool' in weight_index:
                    branch_name='_pool_planes'

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[args.name_base+conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num:
                    select_index = last_select_index
                else:
                    select_index = range(0, orifilter_num)

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                select_index_1 = copy.deepcopy(select_index)

                if orifilter_num != currentfilter_num:
                    logger.info('loading rank from: ' + prefix + str(cov_id) + branch_name + subfix)
                    rank = np.load(prefix + str(cov_id) + branch_name + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                else:
                    select_index = list(range(0, orifilter_num))

                if branch_name == '_n5x5':
                    tmp_select_index = [x+filters[cov_id-2][0]+filters[cov_id-2][1] for x in select_index]
                    cur_last_select_index += tmp_select_index

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                        state_dict[args.name_base+conv_name][index_i][index_j] = \
                            oristate_dict[conv_name][i][j]

        elif name=='pre_layers':

            cnt += 1
            cov_id = cnt

            honey_filter_index = ['.0']  # the index of sketch filter weight
            honey_bn_index = ['.1']  # the index of sketch bn weight

            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_index:

                conv_name = name + weight_index + '.weight'

                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight =state_dict[args.name_base+conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    cur_last_select_index = select_index[:]

                    for index_i, i in enumerate(select_index):
                       state_dict[args.name_base+conv_name][index_i] = \
                            oristate_dict[conv_name][i]#'''

    for name, module in model.named_modules():  # Reassign non sketch weights to the new network
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            if name not in all_honey_conv_name:
                state_dict[args.name_base+name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[args.name_base+name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[args.name_base+name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[args.name_base+name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[args.name_base+name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[args.name_base+name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[args.name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[args.name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_densenet_model(args, logger, model, oristate_dict):

    state_dict = model.state_dict()
    last_select_index = [] #Conv index selected in the previous layer

    cnt=0
    prefix = args.rank_conv_prefix+'/rank_conv_w'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            cov_id = cnt
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[args.name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                rank = np.load(prefix + str(cov_id) + subfix)
                select_index = list(np.argsort(rank)[orifilter_num-currentfilter_num:])  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[args.name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[args.name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[args.name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
                select_index = list(range(0, orifilter_num))

            else:
                select_index = list(range(0, orifilter_num))
                state_dict[args.name_base+name + '.weight'] = oriweight

            if cov_id==1 or cov_id==14 or cov_id==27:
                last_select_index = select_index
            else:
                tmp_select_index = [x+cov_id*12-(cov_id-1)//13*12 for x in select_index]
                last_select_index += tmp_select_index

    model.load_state_dict(state_dict)