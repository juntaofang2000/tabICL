import os
from utils import args
from Dataset import dataloader
from models.runner import supervised, pre_training, linear_probing
from Dataset import load_UEA_data

if __name__ == '__main__':
    # config = args.Initialization(args)
    # # config['data_dir']  =  '/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/UCRArchive_2018'
    # for problem in os.listdir(config['data_dir']):
    #     config['problem'] = problem
    #     print(problem)
    #     Data = dataloader.data_loader(config)

    #     if config['Training_mode'] == 'Pre_Training':
    #         if config['Model_Type'][0] == 'Series2Vec':
    #             best_aggr_metrics_test, all_metrics = pre_training(config, Data)
    #     elif config['Training_mode'] == 'Linear_Probing':
    #         best_aggr_metrics_test, all_metrics = linear_probing(config, Data)
    #     elif config['Training_mode'] == 'Supervised':
    #         best_aggr_metrics_test, all_metrics = supervised(config, Data)

    #     print_str = 'Best Model Test Summary: '
    #     for k, v in best_aggr_metrics_test.items():
    #         print_str += '{}: {} | '.format(k, v)
    #     print(print_str)

    #     with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
    #         for k, v in all_metrics.items():
    #             file.write(f'{k}: {v}\n')
    
    config = {
        'data_dir': '/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/Multivariate_ts',
        'Norm': True,
        'val_ratio': 0.1,
    }

    root_dir = config['data_dir']
    for dataset_name in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        dataset_config = config.copy()
        dataset_config['dataset'] = dataset_name
        print(f'Processing {dataset_name}...')
        _ = load_UEA_data.load(dataset_config)
        print(f'Finished {dataset_name}')