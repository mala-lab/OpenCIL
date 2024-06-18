from opencil.utils import Config

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor, BaseCILFinetunePostprocessor
from .odin_postprocessor import ODINPostprocessor, ODINCILPostprocessor, ODINCILFinetunePostprocessor
from .ebo_postprocessor import EBOPostprocessor, EBOCILPostprocessor, EBOCILFinetunePostprocessor
from .kl_matching_postprocessor import KLMatchingPostprocessor, KLMatchingCILPostprocessor, KLMatchingCILFinetunePostprocessor

# for cil
from .maxlogit_postprocessor import MaxLogitCILPostprocessor
from .react_postprocessor import ReActPostprocessor
from .gen_postprocessor import GENPostprocessor
from .nnguide_postprocessor import NNGuidePostprocessor
from .relation_postprocessor import RelationPostprocessor

def get_postprocessor(config: Config):
    postprocessors = {
        'maxlogit_cil': MaxLogitCILPostprocessor,
        'react_cil': ReActPostprocessor,
        'gen_cil': GENPostprocessor,
        'nnguide_cil': NNGuidePostprocessor,
        'relation_cil': RelationPostprocessor,
        'odin': ODINPostprocessor,
        'odin_cil': ODINCILPostprocessor,
        'odin_cil_finetune': ODINCILFinetunePostprocessor,
        'msp': BasePostprocessor,
        'msp_cil': BaseCILPostprocessor,
        'msp_cil_finetune': BaseCILFinetunePostprocessor,
        'ebo': EBOPostprocessor,
        'ebo_cil': EBOCILPostprocessor,
        'ebo_cil_finetune': EBOCILFinetunePostprocessor,
        'klm': KLMatchingPostprocessor,
        'klm_cil': KLMatchingCILPostprocessor,
        'klm_cil_finetune': KLMatchingCILFinetunePostprocessor,
}
    return postprocessors[config.postprocessor.name](config)
