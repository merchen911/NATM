from argparse import ArgumentParser
import os
from glob import glob

def load_default():
    parser = ArgumentParser()

    ## Default model param
    parser.add_argument('--input_length', default=8, type=int)
    parser.add_argument('--output_length', default=1, type=int)
    parser.add_argument('--seed', default=2022, type=int)

    parser.add_argument('--dataset_list', default=[
                            'ETTm1','ETTm2','ETTh1','ETTh2',
                            'Shunyi','Tiantan', 'Dingling', 'Huairou',
                            'Changping', 'Wanshouxigong', 'Guanyuan', 'Gucheng',
                            'Wanliu', 'Dongsi', 'Nongzhanguan', 'Aotizhongxin',
                            'Solar',
        'Car Traffic',
        'Web Traffic',
        'Electricity'
                        ], type = list)

    parser.add_argument('--dataset_path', default = os.path.join('.','Datasets'), type = str)
    parser.add_argument('--data_path', default=None, type = str)
    parser.add_argument('--dataname', default=None, type = str)
    parser.add_argument('--dataset_name', default=None, type = str)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--training_size', default=.4, type=float)
    parser.add_argument('--multi_thread', default = 3)

    ## Exp variable params
    parser.add_argument('--exp_name', default = None, type = str)
    parser.add_argument('--fold_numb', default = None, type = str)
    parser.add_argument('--gpu_numb', default = None, type = str)
    parser.add_argument('--method', default = None, type = str)

    ## Training params
    parser.add_argument('--prog_bar', default = False, type = bool)
    parser.add_argument('--regression', default = True, type = bool)
    parser.add_argument('--epochs', default = 1000, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--batch_size', default = 128, type = int)
    parser.add_argument('--num_workers', default = 1, type = int)
    parser.add_argument('--ealry_stop_round', default = 10, type = int)
    parser.add_argument('--save_top_k', default = 1, type = int)

    ## Global model params
    parser.add_argument('--dropout', default = 0.1, type = float)

    ## NAM modules params
    parser.add_argument('--nam_hidden_sizes', default = [32], type = list)
    parser.add_argument('--nam_activation', default = 'exu', type = str)
    parser.add_argument('--nam_basis_functions', default = 100, type = int)
    parser.add_argument('--nam_output_bias', default = False, type = bool)
    parser.add_argument('--nam_output_dropout', default = .1, type = float)
    parser.add_argument('--activation', default = 'leaky', type = str)
    parser.add_argument('--shuffle', default = True, type = bool)
    parser.add_argument('--mean', default = .0, type = float)
    parser.add_argument('--std', default = 1., type = float)

    ## Nam regularization techniques
    parser.add_argument('--feature_dropout', default = 0., type = float)
    parser.add_argument('--l2_regularization', default = 0., type = float)
    parser.add_argument('--output_regularization', default = 0., type = float)

    ## LSTM modules param
    parser.add_argument('--lstm_hidden', default = 100, type = int)
    parser.add_argument('--lstm_num_layers', default = 2, type = int)
    parser.add_argument('--lstm_bidirectional', default = True, type = bool)

    ## DNN modules param
    parser.add_argument('--dnn_hiddens', default = [128,128,128], type = list)

    ## Informer modules param
    parser.add_argument('--d_model', default = 64, type = int)
    parser.add_argument('--n_heads', default = 4, type = int)
    parser.add_argument('--d_ff', default = 64, type = int)
    parser.add_argument('--seq_len', default = None, type = int)
    parser.add_argument('--label_len', default = None, type = int)
    parser.add_argument('--pred_len', default = None, type = int)

    config = parser.parse_args([])
    return config




def dataset_name_check(config, dataset_name):
    
    BAQ_name_lists = [i.split('/')[-1][:-4] for i in glob(os.path.join(config.dataset_path,'beijing air quality', '*'))]
    
    if 'ETT' in dataset_name:
        config.data_path = os.path.join(config.dataset_path,'electricity transfer temperature')
        
    elif dataset_name == 'M4':
        config.data_path = os.path.join(config.dataset_path,'M4-wo-nan')
        
    elif dataset_name == 'GAS':
        config.data_path = os.path.join(config.dataset_path,'Gas-sensor-temperature')
        
    elif dataset_name in BAQ_name_lists:
        config.data_path = os.path.join(config.dataset_path,'beijing air quality')
        
    elif dataset_name == 'Solar':
        config.data_path = os.path.join(config.dataset_path,'solar')
        
    elif dataset_name == 'Car Traffic':
        config.data_path = os.path.join(config.dataset_path,'traffic')
        
    elif dataset_name == 'Web Traffic':
        config.data_path = os.path.join(config.dataset_path,'web traffic')
        
    elif dataset_name == 'Electricity':
        config.data_path = os.path.join(config.dataset_path,'electricity load')
        
    return config