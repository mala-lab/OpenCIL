import opencil.utils.comm as comm
import argparse
import numpy as np
import pdb
import logging
import json
import copy
import os.path as osp
import os
import time
from opencil.datasets import get_data_manager, get_cil_ood_dataloader, get_concat_dataloader
from opencil.evaluators import get_evaluator
from opencil.networks import get_network, get_cil_network
from opencil.recorders import get_recorder
from opencil.postprocessors import get_postprocessor
from opencil.trainers import get_il_trainer_pycil
from opencil.utils import setup_logger, update_ood_metrics_by_task, \
                            gen_ood_initialized_metric, log_ood_metric, \
                            update_ood_dict_res, extract_ood_score

from opencil.utils.toolkit import count_parameters
from tqdm import tqdm

from opencil.loggers.exp_logger import MultiLogger

class OODAfterCILPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        logger_ood = MultiLogger(self.config.output_dir, 'ood_output', loggers=['disk'], save_models=False)

        path_save_ood_res_current = osp.join(self.config.output_dir, 'res_ood_current.json')
        path_save_ood_res_history = osp.join(self.config.output_dir, 'res_ood_history.json')
        path_save_ood_output_inspect = osp.join(self.config.output_dir, 'ood_output', 'ood_score_inference') # logger ood create this 


        # init cil ID dataset
        id_data_manager = get_data_manager(self.config)

        # init trainer, need double check from this 
        trainer = get_il_trainer_pycil(self.config)

        # init postprocessor for ood
        postprocessor = get_postprocessor(self.config)

        # init ood evaluator    
        evaluator = get_evaluator(self.config)

        # 
        train_id_loader = []
        val_id_loader = []
        test_id_loader = []

        #
        train_id_data = []
        val_id_data = []
        test_id_data = []

        #
        list_num_test_samples_pre_id = []
        list_num_test_samples_old_id = []
        list_num_test_samples_all_id = []

        # metrics for ood-id accumulating samples
        list_allid_near_ood = []
        list_allid_far_ood = []

        num_task = id_data_manager.nb_tasks

        fpr_all = gen_ood_initialized_metric(num_task)
        auroc_all = gen_ood_initialized_metric(num_task)
        aupr_all = gen_ood_initialized_metric(num_task)

        # ood save dict for current task
        res_ood_dict_current = {'near_ood': {},
                    'far_ood': {}}
        old_network = None

        for task in range(num_task):

            if task >= 1:
                trainer.after_task() # update trainer configuration 
                old_network = trainer.get_oldnetwork()
                old_network.freeze()
                
            # constructing ckpt path by checking which cil trainer, dataset, and network being used
            ckpt_path = os.path.join(self.config.ckpt_path, 'model_ckpt', f'taskid_{str(task)}.pkl')
            trainer.load_checkpoint(id_data_manager, ckpt_path)

            latest_network = trainer.get_network()
            latest_network.freeze()
            

            # check if which network should be used or used both depending on postprocessor type
            # if 'kd' in self.config.postprocessor.name and task >= 1: # kd=>my approach
            #     network = [old_network, latest_network]
            # else:
            #     network = latest_network
            
            if 'kd' in self.config.postprocessor.name:
                network = [old_network, latest_network]
            else:
                network = [None, latest_network]

            # ########### Test accumulating OOD-ID samples ###########

            # get latest train, val, and test data
            latest_id_train_data, _ = trainer.pick_dataloader(id_data_manager, 
                                                             type='latest', 
                                                             mode='train',
                                                             is_ood_process=True)
            
            latest_id_val_data, _ = trainer.pick_dataloader(id_data_manager, 
                                                             type='latest', 
                                                             mode='val') # previously test
            
            latest_id_test_data, _ = trainer.pick_dataloader(id_data_manager, 
                                                             type='latest', 
                                                             mode='test',
                                                             is_ood_process=True)

            train_id_data.append(latest_id_train_data)
            val_id_data.append(latest_id_val_data)
            test_id_data.append(latest_id_test_data)

            # current evaluation only
            print('Processing OOD evaluation on current set of OOD-ID')
            all_id_samples_dict = {'train': train_id_data,
                                        'val': val_id_data,
                                        'test': test_id_data} #ignoring current id samples

            ### construct ood loader dict for current task
            all_ood_loader_dict, _ = get_cil_ood_dataloader(self.config, num_task, task, type='accumulating')

            all_id_loader_dict = get_concat_dataloader(self.config, all_id_samples_dict)
            latest_id_loader = get_concat_dataloader(self.config, {'train': [latest_id_train_data],
                                                                    'val': [latest_id_val_data],
                                                                    'test': [latest_id_test_data]})

            postprocessor.setup(network, latest_id_loader, all_ood_loader_dict)
            res_allid_near_ood, res_allid_far_ood = evaluator.eval_ood(network, all_id_loader_dict, 
                                                                            all_ood_loader_dict, 
                                                                            postprocessor)

            # fpr_all, auroc_all, aupr_all = update_ood_metrics_by_task(fpr_all, auroc_all, aupr_all, 
            #                                                         res_allid_near_ood, res_allid_far_ood,
            #                                                         task, task)
            
            list_allid_near_ood.append(res_allid_near_ood)
            list_allid_far_ood.append(res_allid_far_ood)
            list_num_test_samples_all_id.append(len(all_id_loader_dict['test'].dataset))
                                                                                
            
            # log evaluation result on accumulating id-ood samples
            log_ood_metric(fpr_all, auroc_all, aupr_all, logger_ood, name='all')

            # log evaluation result on per task id-ood samples
            # log_ood_metric(fpr, auroc, aupr, logger_ood, name='per_task')

            evaluator._log_txt_ood_by_task([list_allid_near_ood, list_allid_far_ood],
                                            list_num_test_samples_all_id)

            # save dictionary result

            latest_num_test_id_samples = list_num_test_samples_all_id[-1]
            
            res_ood_dict_current = update_ood_dict_res(res_ood_dict_current, res_allid_near_ood, 
                                            latest_num_test_id_samples, type_ood='near_ood')
            res_ood_dict_current = update_ood_dict_res(res_ood_dict_current, res_allid_far_ood, 
                                            latest_num_test_id_samples, type_ood='far_ood')

            # save ood_score
            extract_ood_score([train_id_data, val_id_data, test_id_data], 
                              self.config, network, postprocessor, evaluator, 
                                path_save_ood_output_inspect, task, num_task)

            # save ood result
            with open(path_save_ood_res_current, 'w') as fp:
                json.dump(res_ood_dict_current, fp, indent=4)

            # if task == num_task - 1:
            #     # Backtracing to previous task set for doing OOD evaluation
            #     print('Backtracing to perform OOD evaluation on growing set of id-ood pairs')
            #     # init dictionary result
            #     res_ood_dict_history = {'near_ood': {}, 'far_ood': {}}

            #     for run_task_idx in tqdm(range(task+1)):
                    
            #         # init dictionary result

            #         # currently at model [task] and evaluate on set [run_task_idx]
            #         history_train_id_data = train_id_data[:run_task_idx+1]
            #         history_val_id_data = val_id_data[:run_task_idx+1]
            #         history_test_id_data = test_id_data[:run_task_idx+1]
                
            #         print('Processing OOD evaluation on accumulating OOD-ID')
            #         history_id_samples_dict = {'train': history_train_id_data,
            #                                     'val': history_val_id_data,
            #                                     'test': history_test_id_data} #ignoring current id samples

            #         ### construct ood loader dict for current task
            #         history_ood_loader_dict, _ = get_cil_ood_dataloader(self.config, num_task, run_task_idx, type='accumulating')

            #         history_id_loader_dict = get_concat_dataloader(self.config, history_id_samples_dict)

            #         # postprocessor.setup(network, history_id_loader_dict, history_ood_loader_dict)
            #         res_historyid_near_ood, res_historyid_far_ood = evaluator.eval_ood(network, history_id_loader_dict, 
            #                                                                         history_ood_loader_dict, 
            #                                                                         postprocessor)                                                                                    

            #         latest_num_test_id_samples = len(history_id_loader_dict['test'].dataset)
                    
            #         res_ood_dict_history = update_ood_dict_res(res_ood_dict_history, res_historyid_near_ood, 
            #                                         latest_num_test_id_samples, type_ood='near_ood')
            #         res_ood_dict_history = update_ood_dict_res(res_ood_dict_history, res_historyid_far_ood, 
            #                                         latest_num_test_id_samples, type_ood='far_ood')
                    
                    
            #     with open(path_save_ood_res_history, 'w') as fp:
            #         json.dump(res_ood_dict_history, fp, indent=4)