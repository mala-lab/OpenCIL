from opencil.utils import Config

from .test_acc_pipeline import TestAccPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_pipeline import TrainPipeline

from .flow_cil_only import CILPipeline
from .flow_ood_after_cil import OODAfterCILPipeline

def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'test_acc': TestAccPipeline,
        'test_ood': TestOODPipeline,
        'cil_only': CILPipeline,
        'ood_after_cil': OODAfterCILPipeline,
    }

    return pipelines[config.pipeline.name](config)
