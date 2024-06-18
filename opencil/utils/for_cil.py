import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp

from tqdm import tqdm
from opencil.datasets import get_concat_dataloader, get_cil_ood_dataloader


def update_ood_metrics_by_task(fpr_dict, auroc_dict, aupr_dict, res_nearood, res_farood, task_id, current_u):

    # near ood
    for data_name in res_nearood.keys():
        [fpr, auroc, aupr_in, _, _, _, _, _, accuracy, num_ood_samples] = res_nearood[data_name][0]
        fpr_dict['nearood'][data_name][task_id, current_u] = fpr
        auroc_dict['nearood'][data_name][task_id, current_u] = auroc
        aupr_dict['nearood'][data_name][task_id, current_u] = aupr_in

    # far ood
    for data_name in res_farood.keys():
        [fpr, auroc, aupr_in, _, _, _, _, _, accuracy, num_ood_samples] = res_farood[data_name][0]
        fpr_dict['farood'][data_name][task_id, current_u] = fpr
        auroc_dict['farood'][data_name][task_id, current_u] = auroc
        aupr_dict['farood'][data_name][task_id, current_u] = aupr_in

    return fpr_dict, auroc_dict, aupr_dict


def gen_ood_initialized_metric(max_task):
    return {'nearood': {
            'cifar10': np.zeros((max_task, max_task)),
            'tin': np.zeros((max_task, max_task))
        },  'farood': {
            'mnist': np.zeros((max_task, max_task)),
            'svhn': np.zeros((max_task, max_task)),
            'texture': np.zeros((max_task, max_task)),
            'places365': np.zeros((max_task, max_task))
        }}


def log_ood_metric(fpr_dict, auroc_dict, aupr_dict, logger, name=None):
    def sub_log(dict_log, logger, type_metric, name_result):
        for type_ood in dict_log.keys():
            for data_name in dict_log[type_ood].keys():
                name_file_log = '{}_{}_{}_{}'.format(type_metric, name_result, type_ood, data_name)
                res = dict_log[type_ood][data_name]

                logger.log_result(res, name=name_file_log, step=None)

    # fpr
    sub_log(fpr_dict, logger, 'fpr@95', name)

    # auroc
    sub_log(auroc_dict, logger, 'auroc', name)

    # aupr
    sub_log(aupr_dict, logger, 'aupr', name)


def update_ood_dict_res(res_ood_dict, res_allid_ood, latest_num_test_id_samples, type_ood=None):
    for data_name in res_allid_ood:
        if data_name not in res_ood_dict[type_ood]:
            res_ood_dict[type_ood][data_name] = {'task1': []}
            cur_task_key = 'task1'
        else:
            latest_task = len(res_ood_dict[type_ood][data_name].keys())

            cur_task_key = 'task'+str(latest_task+1)
            res_ood_dict[type_ood][data_name][cur_task_key] = []

        [fpr, auroc, aupr_in, aupr_out, _, _, _, _, accuracy, num_ood_samples] = res_allid_ood[data_name][0]
        res_ood_dict[type_ood][data_name][cur_task_key] = [fpr, auroc, aupr_in, aupr_out, 
                                                             num_ood_samples, latest_num_test_id_samples]
    return res_ood_dict


def visualize_heatmap(co_occurrence_matrix, save_path, task_id):
    plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference
    sns.heatmap(co_occurrence_matrix, annot=True, cmap='YlOrRd')
    plt.xlabel('cur_model Prediction')  # Label the x-axis
    plt.ylabel('old_model Prediction')  # Label the y-axis
    plt.title(f'Co-occurrence Matrix Task {task_id} Heatmap')  # Add a title to the heatmap

    write_fig_p = osp.join(save_path, f'heatmap_{task_id}.png')
    plt.savefig(write_fig_p)


# save ood score
# Extracting ood score for all samples
def extract_ood_score(data, config, network, postprocessor, evaluator, 
                      path_save, task_id, num_task):
    [train_id_data, val_id_data, test_id_data] = data
    print("Saving ood prediction score")
    ### For incremental id first
    for idx_run_id_data in tqdm(range(len(train_id_data))):
        run_train_data = train_id_data[idx_run_id_data]
        run_val_data = val_id_data[idx_run_id_data]
        run_test_data = test_id_data[idx_run_id_data]
    
        id_samples_dict = {'train': [run_train_data],
                                    'val': [run_val_data],
                                    'test': [run_test_data]}

        ### construct id loader dict
        id_loader_dict = get_concat_dataloader(config, id_samples_dict)

        # extract score of test only
        pred, scores, label = evaluator.extract_scores(network, id_loader_dict['test'], postprocessor)
        
        np.savez(osp.join(path_save, f'task_{task_id}_id_{idx_run_id_data}'),
                    pred=pred,
                    scores=scores,
                    label=label)
    ### construct ood loader dict for current task
    all_ood_loader_dict, _ = get_cil_ood_dataloader(config, num_task, task_id, type='accumulating')

    # add key to my_dict and get score
    ## nearood
    for dataset_name, ood_dl in all_ood_loader_dict['nearood'].items():
        
        pred_near, scores_near, label_near = evaluator.extract_scores(network, ood_dl, postprocessor)
        np.savez(osp.join(path_save, f'task_{task_id}_{dataset_name}'),
                    pred=pred_near,
                    scores=scores_near,
                    label=label_near)

    ## farood
    for dataset_name, ood_dl in all_ood_loader_dict['farood'].items():
        
        pred_far, scores_far, label_far = evaluator.extract_scores(network, ood_dl, postprocessor)
        
        np.savez(osp.join(path_save, f'task_{task_id}_{dataset_name}'),
                    pred=pred_far,
                    scores=scores_far,
                    label=label_far)

            