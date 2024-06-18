import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from opencil.trainers.base_incremental_learning_pycil import BaseLearner
from opencil.networks.incremental_net_pycil import IncrementalNetWithBias


class BiCILearner(BaseLearner):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self._network = IncrementalNetWithBias(
            config, False, bias_correction=True
        )
        self._class_means = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    

    def load_checkpoint(self, data_manager, ckpt_path):
        '''
            The behavior of this function should be the same as incremental train. The only
            difference is that there is no training in this stage. This is because incremental learning
            model grow over the task and the architecture need to grow as well in order to receive the correct 
            pretrained weight
        '''
        print(f"Loading checkpoint from {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        self.data_manager = data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)

        if self._cur_task >= 1:
            self.lamda = self._known_classes / self._total_classes

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')

        self._network.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda")

        self._network.to(device)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task >= 1:
            _, [self.train_loader, self.val_loader] = self.pick_dataloader(data_manager, type='latest', mode='trainval')
            self.lamda = self._known_classes / self._total_classes
        else:
            _, self.train_loader = self.pick_dataloader(data_manager, type='latest', mode='train')

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')
    
        self._log_bias_params()
        self._stage1_training(self.train_loader, self.test_loader)
        if self._cur_task >= 1:
            print("Eval before bias correction")
            cnn_accy, _ = self.eval_task()
            print(cnn_accy)
            self._network.train()

            self._stage2_bias_correction(self.val_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._log_bias_params()

    def get_network(self):
        return self._network
    
    def get_oldnetwork(self):
        return self._old_network

    def pick_dataloader(self, data_manager, type=None, mode=None, is_ood_process=None):
        if type == 'all':
            start = 0
        elif type == 'latest':
            start = self._known_classes

        if mode == 'train':
            dataset = data_manager.get_dataset(
                np.arange(start, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.dataset.train.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        elif mode == 'trainval':
            train_dataset, val_dataset = data_manager.get_dataset_with_split(
                np.arange(start, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
                val_samples_per_class=int(
                    self.config.optimizer.split_ratio * self._memory_size / self._known_classes
                ),
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.dataset.val.batch_size, 
                shuffle=True, 
                num_workers=self.config.num_workers
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.dataset.train.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            
            dataset = [train_dataset, val_dataset]
            dataloader = [train_loader, val_loader]
        
        elif mode == 'val':
            dataset = data_manager.get_dataset(
                np.arange(start, self._total_classes),
                source="val",
                mode="val",
                appendent=self._get_memory(),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.dataset.val.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        elif mode == 'test':
            dataset = data_manager.get_dataset(
                np.arange(start, self._total_classes), source="test", mode="test", ood_eval=is_ood_process
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.dataset.test.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        return dataset, dataloader

    def _run(self, train_loader, test_loader, optimizer, scheduler, stage):
        if stage == 'training':
            prog_bar = tqdm(range(self.config.optimizer.epochs))
        elif stage == 'bias_correction':
            prog_bar = tqdm(range(self.config.optimizer.epochs_bias_correction))
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                if stage == "training":
                    clf_loss = F.cross_entropy(logits, targets)
                    if self._old_network is not None:
                        old_logits = self._old_network(inputs)["logits"].detach()
                        hat_pai_k = F.softmax(old_logits / self.config.optimizer.T, dim=1)
                        log_pai_k = F.log_softmax(
                            logits[:, : self._known_classes] / self.config.optimizer.T, dim=1
                        )
                        distill_loss = -torch.mean(
                            torch.sum(hat_pai_k * log_pai_k, dim=1)
                        )
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss
                elif stage == "bias_correction":
                    loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # debug
                # fa = set(targets.cpu().tolist())
                # set_debug.update(fa)
                

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
                stage,
                self._cur_task,
                epoch,
                self.config.optimizer.epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        """
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        """
        

        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.config.optimizer.lrate, "weight_decay": self.config.optimizer.weight_decay},
            {
                "params": self._network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        optimizer = optim.SGD(
            network_params, lr=self.config.optimizer.lrate, momentum=0.9, weight_decay=self.config.optimizer.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.config.optimizer.milestones, gamma=self.config.optimizer.lrate_decay
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        
        self._run(train_loader, test_loader, optimizer, scheduler, stage="training")

    def _stage2_bias_correction(self, val_loader, test_loader):
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        network_params = [
            {
                "params": self._network.bias_layers[-1].parameters(),
                "lr": self.config.optimizer.lrate,
                "weight_decay": self.config.optimizer.weight_decay,
            }
        ]
        optimizer = optim.SGD(
            network_params, lr=self.config.optimizer.lrate, momentum=0.9, weight_decay=self.config.optimizer.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.config.optimizer.milestones, gamma=self.config.optimizer.lrate_decay
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

        self._run(
            val_loader, test_loader, optimizer, scheduler, stage="bias_correction"
        )

    def _log_bias_params(self):
        logging.info("Parameters of bias layer:")
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            logging.info("{} => {:.3f}, {:.3f}".format(i, param[0], param[1]))
