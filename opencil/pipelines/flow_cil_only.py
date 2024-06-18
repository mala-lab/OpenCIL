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
from opencil.utils import setup_logger
from opencil.utils.toolkit import count_parameters
from tqdm import tqdm

from opencil.loggers.exp_logger import MultiLogger

class CILPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        path_save_cil_res = osp.join(self.config.output_dir, 'cil_output')

        os.makedirs(path_save_cil_res, exist_ok=True)
        
        # init cil ID dataset
        id_data_manager = get_data_manager(self.config)

        # init trainer, need double check from this 
        trainer = get_il_trainer_pycil(self.config)

        num_task = id_data_manager.nb_tasks

        for task in range(num_task):

            if task >= 1:
                trainer.after_task() # update trainer configuration 

            logging.info("All params: {}".format(count_parameters(trainer._network)))
            logging.info(
                "Trainable params: {}".format(count_parameters(trainer._network, True))
            )
            print("Training task:", str(task+1))
            trainer.incremental_train(id_data_manager)
            
            print("Eval")
            cnn_accy, _ = trainer.eval_task()

            path_save_cil_res_each_task = osp.join(path_save_cil_res, 'res_cil_after_task'+str(task)+'.json')
            # save result to disk
            with open(path_save_cil_res_each_task, 'w') as fp:
                json.dump(cnn_accy, fp, indent=4)

            print(cnn_accy)
            # save checkpoint CIL
            path_folder_save = osp.join(self.config.output_dir, 'model_ckpt')
            os.makedirs(path_folder_save, exist_ok=True)
            path_ckpt_save = osp.join(path_folder_save, 'taskid')

            trainer.save_checkpoint(path_ckpt_save)