import logging
import os.path as osp
import os
import numpy as np
import pdb
import copy
import time
import faiss.contrib.torch_utils
from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from opencil.trainers.base_incremental_learning_pycil import BaseLearner
from opencil.networks.incremental_net_pycil import FinetuneNet_foster, FinetuneNet_bic, FinetuneNet_icarl, FinetuneNet_wa
from opencil.utils.toolkit import count_parameters, target2onehot, tensor2numpy

class FinetuneVOSLearner(BaseLearner):
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




        # a bunch of constants or hyperparams
        self.sample_number = self.config.trainer.sample_number
        self.sample_from = self.config.trainer.sample_from
        self.select = self.config.trainer.select

        try:
            self.penultimate_dim = self._network.feature_dim
        except AttributeError:
            self.penultimate_dim = self._network.module.feature_dim

        # self.n_cls = self._total_classes
        self.n_cls = self._total_classes - self._known_classes

        self.data_dict = torch.zeros(self.n_cls, self.sample_number,
                                self.penultimate_dim).cuda()
        
        self.number_dict = {}
        for i in range(self.n_cls):
            self.number_dict[i] = 0

        eye_matrix = torch.eye(self.penultimate_dim, device='cuda')
        
        if self._cur_task == 0:
            prog_bar = tqdm(range(self.config.optimizer.num_epochs))
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0
                for i, (train_datas, train_targets) in enumerate(train_loader):
                    train_datas, train_targets = train_datas.to(self._device), train_targets.to(self._device)
                    
                    out = self._network(train_datas, return_feature = True)
                    logits, features = out["aux_logits"], out["features"]

                    loss = F.cross_entropy(logits, train_targets)

                    sum_temp = 0
                    for index in range(self.n_cls):
                        sum_temp += self.number_dict[index]

                    
                    if sum_temp == self.n_cls * self.sample_number:
                        # maintaining an ID data queue for each class.
                        target_numpy = train_targets.cpu().data.numpy() - self._known_classes
                        for index in range(len(train_targets)):
                            dict_key = target_numpy[index]
                            self.data_dict[dict_key] = torch.cat(
                                (self.data_dict[dict_key][1:],
                                features[index].detach().view(1, -1)), 0)

                        for index in range(self.n_cls):
                            if index == 0:
                                X = self.data_dict[index] - self.data_dict[index].mean(
                                    0)
                                mean_embed_id = self.data_dict[index].mean(0).view(
                                    1, -1)
                            else:
                                X = torch.cat((X, self.data_dict[index] -
                                            self.data_dict[index].mean(0)), 0)
                                mean_embed_id = torch.cat(
                                    (mean_embed_id, self.data_dict[index].mean(0).view(
                                        1, -1)), 0)
                        
                        # Standard Gaussian distribution
                        temp_precision = torch.mm(X.t(), X) / len(X)
                        temp_precision += 0.0001 * eye_matrix
                        for index in range(self.n_cls):
                            new_dis = MultivariateNormal(
                                loc=mean_embed_id[index],
                                covariance_matrix=temp_precision)
                            negative_samples = new_dis.rsample(
                                (self.sample_from, ))
                            prob_density = new_dis.log_prob(negative_samples)
                            cur_samples, index_prob = torch.topk(
                                -prob_density, self.select)
                            if index == 0:
                                ood_samples = negative_samples[index_prob]
                            else:
                                ood_samples = torch.cat(
                                    (ood_samples, negative_samples[index_prob]), 0)                       

                        if len(ood_samples) != 0:
                            ood_logit = self._network.aux_fc(ood_samples)["logits"]
                            Ec_in = -torch.logsumexp(logits, dim=1)
                            Ec_out = -torch.logsumexp(ood_logit, dim=1)             
                            loss += 0.1*(torch.pow(F.relu(Ec_in-self.config.trainer.m_in), 2).mean() + (torch.pow(F.relu((self.config.trainer.m_out - Ec_out)), 2)).mean())
 
                    else:
                        target_numpy = train_targets.cpu().data.numpy() - self._known_classes
                        for index in range(len(train_targets)):
                            dict_key = target_numpy[index]
                            if self.number_dict[dict_key] < self.sample_number:
                                self.data_dict[dict_key][self.number_dict[
                                    dict_key]] = features[index].detach()
                                self.number_dict[dict_key] += 1


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
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

                    out = self._network(train_datas, return_feature = True)
                    logits, features = out["aux_logits"], out["features"]

                    loss = F.cross_entropy(logits, train_targets)

                    sum_temp = 0
                    for index in range(self.n_cls):
                        sum_temp += self.number_dict[index]

                    
                    if sum_temp == self.n_cls * self.sample_number:
                        # maintaining an ID data queue for each class.
                        target_numpy = train_targets.cpu().data.numpy() - self._known_classes
                        for index in range(len(train_targets)):
                            dict_key = target_numpy[index]
                            self.data_dict[dict_key] = torch.cat(
                                (self.data_dict[dict_key][1:],
                                features[index].detach().view(1, -1)), 0)

                        for index in range(self.n_cls):
                            if index == 0:
                                X = self.data_dict[index] - self.data_dict[index].mean(
                                    0)
                                mean_embed_id = self.data_dict[index].mean(0).view(
                                    1, -1)
                            else:
                                X = torch.cat((X, self.data_dict[index] -
                                            self.data_dict[index].mean(0)), 0)
                                mean_embed_id = torch.cat(
                                    (mean_embed_id, self.data_dict[index].mean(0).view(
                                        1, -1)), 0)
                        
                        # Standard Gaussian distribution
                        temp_precision = torch.mm(X.t(), X) / len(X)
                        temp_precision += 0.0001 * eye_matrix
                        for index in range(self.n_cls):
                            new_dis = MultivariateNormal(
                                loc=mean_embed_id[index],
                                covariance_matrix=temp_precision)
                            negative_samples = new_dis.rsample(
                                (self.sample_from, ))
                            prob_density = new_dis.log_prob(negative_samples)
                            cur_samples, index_prob = torch.topk(
                                -prob_density, self.select)
                            if index == 0:
                                ood_samples = negative_samples[index_prob]
                            else:
                                ood_samples = torch.cat(
                                    (ood_samples, negative_samples[index_prob]), 0)  

                        if len(ood_samples) != 0:
                            ood_logit = self._network.aux_fc(ood_samples)["logits"]
                            Ec_in = -torch.logsumexp(logits, dim=1)
                            Ec_out = -torch.logsumexp(ood_logit, dim=1)             
                            loss += 0.1*(torch.pow(F.relu(Ec_in-self.config.trainer.m_in), 2).mean() + (torch.pow(F.relu((self.config.trainer.m_out - Ec_out)), 2)).mean())
 
                    else:
                        target_numpy = train_targets.cpu().data.numpy() - self._known_classes
                        for index in range(len(train_targets)):
                            dict_key = target_numpy[index]
                            if self.number_dict[dict_key] < self.sample_number:
                                self.data_dict[dict_key][self.number_dict[
                                    dict_key]] = features[index].detach()
                                self.number_dict[dict_key] += 1

                    val_beta = 0.002
                    mixed_size = val_datas.shape[0] if train_datas.shape[0] >= val_datas.shape[0] else train_datas.shape[0]
                    mixed_val_data = val_beta * train_datas[:mixed_size, :] + (1 - val_beta) * val_datas[:mixed_size, :]
                    mixed_val_logits = self._network(mixed_val_data)["aux_logits"]
                    val_targets = val_targets[:mixed_val_logits.shape[0]]
                    loss += F.cross_entropy(mixed_val_logits, val_targets) 
                    Ec_val_in = -torch.logsumexp(mixed_val_logits, dim=1)                       
                    loss += 0.1 * torch.pow(F.relu(Ec_val_in-self.config.trainer.m_in), 2).mean()


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
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



def generate_outliers(ID,
                      input_index,
                      negative_samples,
                      ID_points_num=2,
                      K=20,
                      select=1,
                      cov_mat=0.1,
                      sampling_ratio=1.0,
                      pic_nums=30,
                      depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[0],
                                int(normed_data.shape[0] * sampling_ratio),
                                replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat(
        [ID[i:i + 1].repeat(length, 1) for i in minD_idx])
    negative_sample_cov = cov_mat * negative_samples.cuda().repeat(pic_nums, 1)
    negative_sample_list = negative_sample_cov + data_point_list
    point = KNN_dis_search_distance(negative_sample_list, index, K,
                                    ID_points_num, length, depth)

    index.reset()
    return point


def KNN_dis_search_distance(target,
                            index,
                            K=50,
                            num_points=10,
                            length=2000,
                            depth=342):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    # Normalize the features
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    # target_new = target.view(length, -1, depth)
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i * length + minD_idx[:, i])
    return target[torch.cat(point_list)]


def KNN_dis_search_decrease(
    target,
    index,
    K=50,
    select=1,
):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    # Normalize the features
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    return minD_idx, k_th_distance