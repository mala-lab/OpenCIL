from opencil.utils import Config

from .base_evaluator import BaseEvaluator
from .ood_evaluator import OODEvaluator
from .ood_cil_evaluator import OODCILEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'ood_cil': OODCILEvaluator
    }
    return evaluators[config.evaluator.name](config)
