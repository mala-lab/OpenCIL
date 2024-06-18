import csv
import os
from typing import Dict, List
from sklearn import metrics 

import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import copy
import pdb
from torch.utils.data import DataLoader

from opencil.postprocessors import BasePostprocessor
from opencil.utils import Config
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist



from .base_cil_evaluator import BaseCILEvaluator
from .metrics import compute_all_metrics
from .separab_draw import *
from .weight_analysis_draw import *

def construct_task_id_pred_histogram(ood_pred, cur_task_id, cp_task):
    unique_count = np.unique(ood_pred, return_counts=True)[1]
    his = {}
    for i in range(cur_task_id+1):
        start = i*cp_task
        end = (i+1)*cp_task
        his[i] = np.sum(unique_count[start:end])
    return his

class OODCILEvaluator(BaseCILEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODCILEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
    
    def set_task_id(self, task_id):
        self.task_id = task_id

    def eval_ood(self, nets, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        [net1, net2] = nets
        if type(net1) is dict:
            for subnet in net1.values():
                subnet.eval()
        else:
            if net1 is not None:
                net1.eval()

        if type(net2) is dict:
            for subnet in net2.values():
                subnet.eval()
        else:
            if net2 is not None:
                net2.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        # print(f'Performing inference on {dataset_name} dataset...', flush=True)

        # if self.config.postprocessor.APS_mode:
        #     self.hyperparam_search(net, id_data_loader['val'],
        #                            ood_data_loaders['val'], postprocessor)

        if net1 is None:
            net_pass =  net2
        else:
            net_pass = [net1, net2]

        id_pred, id_conf, id_gt = postprocessor.inference(net_pass, id_data_loader['test'])
        # if self.config.recorder.save_scores:
        #     self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load nearood data and compute ood metrics
        near_ood_metrics = self._eval_ood(net_pass, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')
        # load farood data and compute ood metrics
        far_ood_metrics = self._eval_ood(net_pass, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

        return near_ood_metrics, far_ood_metrics

    def _eval_ood(self,
                  net,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        # print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        metrics_dict_list = {}
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)
            
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])


            # print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)

            if dataset_name not in metrics_dict_list:
                metrics_dict_list[dataset_name] = []

            # add number of ood samples to the end of this list
            ood_metrics.append(len(ood_dl.dataset))
            metrics_dict_list[dataset_name].append(ood_metrics)

            metrics_list.append(ood_metrics)

        # print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

        return metrics_dict_list

    def extract_scores(self, nets, data_loader, postprocessor):
        [net1, net2] = nets
        if type(net1) is dict:
            for subnet in net1.values():
                subnet.eval()
        else:
            if net1 is not None:
                net1.eval()

        if type(net2) is dict:
            for subnet in net2.values():
                subnet.eval()
        else:
            if net2 is not None:
                net2.eval()

        if net1 is None:
            net_pass =  net2
        else:
            net_pass = [net1, net2]

        # pred, conf, gt = postprocessor.inference(
        #     net_pass, data_loader, output=True)
        pred, conf, gt = postprocessor.inference(
            net_pass, data_loader)
        
        return pred, conf, gt


    def _log_txt_ood_by_task(self, list_allid_metrics, list_num_test_samples_all_id):
        txt_path = os.path.join(self.config.output_dir, 'log_incremental_ood.txt')

        with open(txt_path, 'w') as f:
            dict_loop = {'near ood': 0, 'far ood': 1}

            data_mean_fpr, data_mean_auroc, data_mean_aupr_in, data_mean_aupr_out, data_mean_accuracy = 0, 0, 0, 0, 0

            for split_type in dict_loop:
                list_ood_metrics = dict_loop[split_type]
                f.write(f'OOD results of {split_type} data\n')
                
            
                for each_data in list_allid_metrics[dict_loop[split_type]][0]:
                    
                    if len(list_allid_metrics[0]) == 0:
                        continue
                    f.write(f'+OOD Dataset: {each_data} vs All ID samples\n')
                    chosen_id_metrics = list_allid_metrics
                    chosen_split_type = dict_loop[split_type]
                    list_ood_metrics = chosen_id_metrics[chosen_split_type]

                    mean_fpr, mean_auroc, mean_aupr_in, mean_aupr_out, mean_accuracy = 0, 0, 0, 0, 0

                    for each_task_idx in range(len(list_ood_metrics)):
                        write_head_str = f'--Task {each_task_idx+1}-->'
                        metrics = list_ood_metrics[each_task_idx][each_data][0] # only start from task 2 for old and all id samples
                        
                        [fpr, auroc, aupr_in, aupr_out, _, _, _, _, accuracy, num_ood_samples] = metrics

                        mean_fpr += fpr
                        mean_auroc += auroc
                        mean_aupr_in += aupr_in
                        mean_aupr_out += aupr_out
                        mean_accuracy += accuracy

                        num_id_samples = list_num_test_samples_all_id[each_task_idx]
                        # write_content = {
                        #     'FPR@95': '{:.2f}'.format(100 * fpr),
                        #     'AUROC': '{:.2f}'.format(100 * auroc),
                        #     'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
                        #     'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
                        #     'CCR_4': '{:.2f}'.format(100 * ccr_4),
                        #     'CCR_3': '{:.2f}'.format(100 * ccr_3),
                        #     'CCR_2': '{:.2f}'.format(100 * ccr_2),
                        #     'CCR_1': '{:.2f}'.format(100 * ccr_1),
                        #     'ACC': '{:.2f}'.format(100 * accuracy)
                        # }

                        write_content = {
                            'FPR@95': '{:.2f}'.format(100 * fpr),
                            'AUROC': '{:.2f}'.format(100 * auroc),
                            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
                            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
                            'ACC': '{:.2f}'.format(100 * accuracy),
                            'NumOODSamples': num_ood_samples,
                            'NumIDSamples': num_id_samples
                        }

                        for score_name in write_content:
                            write_head_str += f'{score_name}: {str(write_content[score_name])},'

                        f.write(write_head_str)
                        f.write('\n')
                    
                        # f.write(f'{"="*20}\n')

                    data_mean_fpr += mean_fpr
                    data_mean_auroc += mean_auroc
                    data_mean_aupr_in += mean_aupr_in
                    data_mean_aupr_out += mean_aupr_out
                    data_mean_accuracy += mean_accuracy
                    
                    write_head_str = f'--Task mean-->'
                    write_content = {
                        'mean_FPR@95': '{:.2f}'.format(100 * mean_fpr / (each_task_idx+1)),
                        'mean_AUROC': '{:.2f}'.format(100 * mean_auroc / (each_task_idx+1)),
                        'mean_AUPR_IN': '{:.2f}'.format(100 * mean_aupr_in / (each_task_idx+1)),
                        'mean_AUPR_OUT': '{:.2f}'.format(100 * mean_aupr_out / (each_task_idx+1)),
                        'mean_ACC': '{:.2f}'.format(100 * mean_accuracy / (each_task_idx+1))
                    }

                    for score_name in write_content:
                        write_head_str += f'{score_name}: {str(write_content[score_name])},'

                    f.write(write_head_str)
                    f.write('\n')

                    f.write('\n')

                f.write(f'{"*"*20}\n')

            write_head_str = f'--datasets mean-->'
            write_content = {
                'mean_FPR@95': '{:.2f}'.format(100 * data_mean_fpr / (6 * (each_task_idx+1))),
                'mean_AUROC': '{:.2f}'.format(100 * data_mean_auroc / (6 * (each_task_idx+1))),
                'mean_AUPR_IN': '{:.2f}'.format(100 * data_mean_aupr_in / (6 * (each_task_idx+1))),
                'mean_AUPR_OUT': '{:.2f}'.format(100 * data_mean_aupr_out / (6 * (each_task_idx+1))),
                'mean_ACC': '{:.2f}'.format(100 * data_mean_accuracy / (6 * (each_task_idx+1)))
            }
            for score_name in write_content:
                write_head_str += f'{score_name}: {str(write_content[score_name])},'

            f.write(write_head_str)
            f.write('\n')
            
    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, accuracy] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'CCR_4': '{:.2f}'.format(100 * ccr_4),
            'CCR_3': '{:.2f}'.format(100 * ccr_3),
            'CCR_2': '{:.2f}'.format(100 * ccr_2),
            'CCR_1': '{:.2f}'.format(100 * ccr_1),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f},'.format(
            ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
              end=' ',
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)
        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader)

            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results