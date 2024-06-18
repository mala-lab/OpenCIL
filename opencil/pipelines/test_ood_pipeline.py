import time
import torch

from opencil.datasets import get_dataloader, get_ood_dataloader
from opencil.evaluators import get_evaluator
from opencil.networks import get_network
from opencil.postprocessors import get_postprocessor
from opencil.utils import setup_logger


class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        
        ## check to see if loading directly from facebook research ckpt or local pretrained checkpoint
        if 'fb_research' in self.config.network.checkpoint:
            net = torch.load(self.config.network.checkpoint)
        else:
            net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                         postprocessor)
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
