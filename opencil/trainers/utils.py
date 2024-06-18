import argparse
from torch.utils.data import DataLoader

from opencil.utils import Config
from .base_trainer import BaseTrainer

from .incremental_foster_pycil import FosterILearner
from .incremental_bic_pycil import BiCILearner
from .incremental_icarl_pycil import iCaRLILearner
from .incremental_wa_pycil import WAILearner
from .incremental_finetune_ber_pycil import FinetuneBERLearner
from .incremental_finetune_logitnorm_pycil import FinetuneLogitNormLearner
from .incremental_finetune_t2fnorm_pycil import FinetuneT2FNormLearner
from .incremental_finetune_npos_pycil import FinetuneNPOSLearner
from .incremental_finetune_vos_pycil import FinetuneVOSLearner
from .incremental_finetune_augmix_pycil import FinetuneAUGMIXLearner
from .incremental_finetune_regmix_pycil import FinetuneREGMIXLearner

def get_trainer(net, train_loader: DataLoader, config: Config):
    if type(train_loader) is DataLoader:
        trainers = {
            'base': BaseTrainer,
        }
        return trainers[config.trainer.name](net, train_loader, config)

def get_il_trainer_pycil(config: Config):

    cil_trainers = {
        'incremental_finetune_ber': FinetuneBERLearner,
        'incremental_finetune_logitnorm': FinetuneLogitNormLearner,
        'incremental_finetune_t2fnorm': FinetuneT2FNormLearner,
        'incremental_finetune_npos': FinetuneNPOSLearner,   
        'incremental_finetune_vos': FinetuneVOSLearner,        
        'incremental_finetune_augmix': FinetuneAUGMIXLearner,  
        'incremental_finetune_regmix': FinetuneREGMIXLearner,
        'incremental_foster': FosterILearner,
        'incremental_bic':BiCILearner,
        'incremental_icarl': iCaRLILearner,
        'incremental_wa': WAILearner
    }

    return cil_trainers[config.trainer.name](config)
