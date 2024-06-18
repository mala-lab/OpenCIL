import logging
import os.path as osp
import os
import numpy as np
import pdb
import copy
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from opencil.trainers.base_incremental_learning_pycil import BaseLearner
from opencil.networks.incremental_net_pycil import FinetuneNet_foster, FinetuneNet_bic, FinetuneNet_icarl, FinetuneNet_wa
from opencil.utils.toolkit import count_parameters, target2onehot, tensor2numpy

# https://github.com/FrancescoPinto/RegMixup/blob/main/models/regmixup.py
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda."""

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FinetuneREGMIXLearner(BaseLearner):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        finetune_trainers = {
            'incremental_finetune_foster': FinetuneNet_foster,
            'incremental_finetune_bic': FinetuneNet_bic,
            'incremental_finetune_icarl': FinetuneNet_icarl,
            'incremental_finetune_wa': FinetuneNet_wa
        }

        self._network = finetune_trainers["incremental_finetune_{}".format(config.cilmethod)](self.config, False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def eval_task(self):
        y_pred, y_true = self._eval_cnn_finetune(self.test_loader) # should be replaced with test loader
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

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
        self._network.update_fc(self._total_classes, self.config)

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')

        self._network.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda")

        self._network.to(device)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def set_mode_to_train(self, model):
        print('ALl parameters requiring grad:'+'\n')    
        for name, p in model.named_parameters():
            if 'aux_fc' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
            if p.requires_grad:
                print(name)

        for name, m in self._network.named_modules():
            if 'aux_fc' in name:
                m.train()
            else:
                m.eval()
            if m.training:
                print(name)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes, self.config)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        ckpt_path = "./results/results_cil_only/incremental_{}_{}_{}_increment_pt{}/model_ckpt/taskid_{}.pkl"\
            .format(self.config.cilmethod, self.config.dataset.name, self.config.network.name, self.config.increment, self._cur_task)
        checkpoint = torch.load(ckpt_path)        
        if self.config.cilmethod == "foster":
            self._network.load_state_dict(checkpoint['main_net_model_state_dict'], strict=False)
            # self._network.load_state_dict(checkpoint['main_net_model_state_dict'])
        else:
            self._network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # self._network.load_state_dict(checkpoint['model_state_dict'])

    
        self._network.aux_fc.weight.data = copy.deepcopy(self._network.fc.weight.data)
        self._network.aux_fc.bias.data = copy.deepcopy(self._network.fc.bias.data)

        
        # train loader
        if self._cur_task >= 1:
            _, [self.train_loader, self.val_loader] = self.pick_dataloader(data_manager, type='latest', mode='train_divided')

        else:
            _, self.train_loader = self.pick_dataloader(data_manager, type='latest', mode='train')
            self.val_loader = None

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.val_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

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
        elif mode == 'train_divided':
            train_dataset, val_dataset = data_manager.get_divided_dataset(
                np.arange(start, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
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

    def _train(self, train_loader, val_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.optimizer.num_epochs)

        self.set_mode_to_train(self._network)
        
        if self._cur_task == 0:
            prog_bar = tqdm(range(self.config.optimizer.num_epochs))
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0
                for i, (train_datas, train_targets) in enumerate(train_loader):
                    train_datas, train_targets = train_datas.to(self._device), train_targets.to(self._device)

                    # mixup operation
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(train_datas, train_targets, self.config.trainer.alpha)
                    targets_a = torch.cat([train_targets, part_y_a])
                    targets_b = torch.cat([train_targets, part_y_b])
                    train_datas = torch.cat([train_datas, mixup_x], dim=0)

                    # forward
                    train_logits = self._network(train_datas)["aux_logits"]
                    loss = regmixup_criterion(F.cross_entropy, train_logits, targets_a,
                                            targets_b, lam)


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(train_logits, dim=1)
                    correct += preds[:train_targets.shape[0]].eq(train_targets.expand_as(preds[:train_targets.shape[0]])).cpu().sum()
                    total += len(train_targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy_finetune(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.config.optimizer.num_epochs,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.config.optimizer.num_epochs,
                        losses / len(train_loader),
                        train_acc,
                    )
                prog_bar.set_description(info)
            logging.info(info)
                
        else:
            prog_bar = tqdm(range(self.config.optimizer.num_epochs))
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0
                val_iterator = iter(val_loader)
                for i, (train_datas, train_targets) in enumerate(train_loader):
                    try:
                        val_datas, val_targets = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        val_datas, val_targets = next(val_iterator)

                    train_datas, train_targets = train_datas.to(self._device), train_targets.to(self._device)
                    val_datas, val_targets = val_datas.to(self._device), val_targets.to(self._device)

                    # mixup operation
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(train_datas, train_targets, self.config.trainer.alpha)
                    targets_a = torch.cat([train_targets, part_y_a])
                    targets_b = torch.cat([train_targets, part_y_b])
                    train_datas = torch.cat([train_datas, mixup_x], dim=0)

                    # forward
                    train_logits = self._network(train_datas)["aux_logits"]
                    loss = regmixup_criterion(F.cross_entropy, train_logits, targets_a,
                                            targets_b, lam)

                    # mixup operation
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(val_datas, val_targets, self.config.trainer.alpha)
                    targets_a = torch.cat([val_targets, part_y_a])
                    targets_b = torch.cat([val_targets, part_y_b])
                    val_datas = torch.cat([val_datas, mixup_x], dim=0)

                    # forward
                    val_logits = self._network(val_datas)["aux_logits"]
                    loss += regmixup_criterion(F.cross_entropy, val_logits, targets_a,
                                            targets_b, lam)


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(train_logits, dim=1)
                    correct += preds[:train_targets.shape[0]].eq(train_targets.expand_as(preds[:train_targets.shape[0]])).cpu().sum()
                    total += len(train_targets)              

                
                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy_finetune(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.config.optimizer.num_epochs,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.config.optimizer.num_epochs,
                        losses / len(train_loader),
                        train_acc,
                    )
                prog_bar.set_description(info)
            logging.info(info)

