import logging
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
from opencil.networks.incremental_net_pycil import FOSTERNet
from opencil.utils.toolkit import count_parameters, target2onehot, tensor2numpy

from opencil.networks.incremental_net_pycil import FOSTERNet
# Please refer to https://github.com/G-U-N/ECCV22-FOSTER for the full source code to reproduce foster.


class FosterILearner(BaseLearner):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._network = FOSTERNet(self.config, False)
        self._snet = None
        self.beta1 = self.config.trainer["beta1"]
        self.beta2 = self.config.trainer["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = self.config.trainer["is_teacher_wa"]
        self.is_student_wa = self.config.trainer["is_student_wa"]
        self.lambda_okd = self.config.trainer["lambda_okd"]
        self.wa_value = self.config.trainer["wa_value"]
        self.oofc = self.config.trainer["oofc"].lower()

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def save_checkpoint(self, filename):
        # special checkpoint saving for foster
        print("Saving checkpoint with fosternet")
        self._network.cpu()

        state_dict_mainnet = self._network.state_dict()
        state_dict_snet = None

        if self._snet is not None:
            if hasattr(self._snet, "module"):
                state_dict_snet = self._snet.module.cpu().state_dict()
            else:
                state_dict_snet = self._snet.cpu().state_dict()
        save_dict = {
            "tasks": self._cur_task,
            "main_net_model_state_dict": state_dict_mainnet,
            "s_net_model_state_dict": state_dict_snet
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))


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

        if self._cur_task > 1:
            self._network = self._snet

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes, self.config)
        self._network_module_ptr = self._network

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')
        
        ### treat this block of code as the model after training
        self._network.load_state_dict(checkpoint['main_net_model_state_dict'])
        device = torch.device("cuda")

        self._network.to(self._device)
        # foster require an s net
        if checkpoint['s_net_model_state_dict'] is not None:
            # init snet architecture 
            self._snet = FOSTERNet(self.config, False)
            self._snet.update_fc(self._total_classes, self.config)

            self._snet.load_state_dict(checkpoint['s_net_model_state_dict'])
            self._snet.to(self._device)
        ###

        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes, self.config)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        # train loader
        _, self.train_loader = self.pick_dataloader(data_manager, type='latest', mode='train')

        # test loader
        _, self.test_loader = self.pick_dataloader(data_manager, type='all', mode='test')

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
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

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.config.optimizer["init_lr"],
                weight_decay=self.config.optimizer.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.config.optimizer.init_epochs
            )

            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.config.optimizer.lr,
                momentum=0.9,
                weight_decay=self.config.optimizer.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.config.optimizer.boosting_epochs
            )
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._feature_boosting(train_loader, test_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.config.optimizer.init_epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.config.optimizer.boosting_epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)

                #### debug
                # print(inputs.shape)
                if inputs.shape[0] == 1:
                    pdb.set_trace()

                outputs = self._network(inputs)
                logits, fe_logits, old_logits = (
                    outputs["logits"],
                    outputs["fe_logits"],
                    outputs["old_logits"].detach(),
                )
                loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits, targets)
                loss_kd = self.lambda_okd * _KD_loss(
                    logits[:, : self._known_classes], old_logits, self.config.optimizer.T
                )
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.boosting_epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.boosting_epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self.config, False)
        self._snet.update_fc(self._total_classes, self.config)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.config.optimizer.lr,
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.config.optimizer.compression_epochs
        )
        self._network.eval()
        prog_bar = tqdm(range(self.config.optimizer.compression_epochs))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                dark_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                loss_dark = self.BKD(dark_logits, logits, self.config.optimizer.T)
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.compression_epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.config.optimizer.compression_epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.config.dataset.name == "cifar100_cil":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
