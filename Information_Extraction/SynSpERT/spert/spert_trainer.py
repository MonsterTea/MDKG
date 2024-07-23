import argparse
import math
import os
import numpy as np
import csv
from scipy.special import entr

import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from spert import models
from spert import sampling
from spert import util  ##DKS
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss
from tqdm import tqdm
from spert.trainer import BaseTrainer
from transformers import BertConfig
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace, config: BertConfig):
        super().__init__(args, config)

        # byte-pair encoding
        #DKS: Commented for now
        print("################ ",args.tokenizer_path)
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)
        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def _load_pretrained_model(self, input_reader: BaseInputReader):
        # create model
        model_class = models.get_model(self.args.model_type) 
        #(Above) self.args.model_type = "syn_spert", model_class = 'SpERT'

        # load model
        #config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        config = self.config   #DKS
        util.check_version(config, model_class, self.args.model_path)

        config.spert_version = model_class.VERSION
        print("**** Calling model_class.from_pretrained(): TYPE: ", self.args.model_type, "****")
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            cache_dir=self.args.cache_path,
                                            use_pos=self.args.use_pos,  
                                            #pos_embedding=self.args.pos_embedding,
                                            use_entity_clf=self.args.use_entity_clf
                                            )
        print("Model type = ", type(model))

        return model


    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        
        train_dataset = input_reader.get_dataset(train_label)

        
        
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model = self._load_pretrained_model(input_reader)
        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        
        model.to(self._device)



        for name, para in model.named_parameters():
            model.state_dict()[name][:] += (torch.rand(para.size()).to(self._device) - 0.5) * self.args.noise_lambda * torch.std(
                para).to(self._device)

        def get_layers(model):
            layers = []

            def unfold_layer(model):
                layer_list = list(model.named_children())
                for item in layer_list:
                    module = item[1]
                    sublayer = list(module.named_children())
                    sublayer_num = len(sublayer)

                    if sublayer_num == 0:
                        layers.append(module)
                    elif isinstance(module, nn.Module):
                        unfold_layer(module)


            unfold_layer(model)
            return layers

        for layer in get_layers(model.bert.encoder.layer[-1]):
            if isinstance(layer, (nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

        for layer in get_layers(model.bert.pooler):
            if isinstance(layer, (nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=True)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        best_model = None #DKS
        best_rel_f1_micro=0
        best_epoch=0
        model_saved = False

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets [DKS]
            if not args.final_eval or (epoch == args.epochs - 1):
               # self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
               ner_f1_micro,rel_f1_micro = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
               
               #DKS: check if model is best
               if rel_f1_micro > best_rel_f1_micro:
                   best_rel_f1_micro=rel_f1_micro
                   best_model=model
                   best_epoch=epoch+1
                   extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                   self._save_model(self._save_path, best_model, self._tokenizer, 0,
                       optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                       include_iteration=False, name='best_model')
                   model_saved=True
                   #break
               
            
        sys.exit(0)
        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')
        
        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()
        

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS

        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model = self._load_pretrained_model(input_reader)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)
 
        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()
        
        #print("*************** train_dataset = ", dataset)
        #sys.exit(-1)
        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            
            # forward step
            
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'],
                                              dephead= batch['dephead'], deplabel =batch['deplabel'], pos= batch['pos'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
          epoch: int = 0, updates_epoch: int = 0, iteration: int = 0 ):
    self._logger.info("Evaluate: %s" % dataset.label)
    Names = 'agu1'  # Set the name for saving files

    if isinstance(model, DataParallel):
        # currently no multi GPU support during evaluation
        model = model.module

    # create evaluator
    evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                          self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                          self._examples_path, self.args.example_count, epoch, dataset.label)

    # create data loader
    dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                             num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

    with torch.no_grad():
        model.eval()
        n = 9  # Number of classes

        prediction_entropy_relation = np.ones(1,)
        prediction_entropy_entities = np.ones(1,)
        sentence_size = np.ones(1, )
        prediction_label_all = np.ones(n,)
        pooler_output_all = torch.ones(1, 768).to(device='cuda')

        # iterate batches
        total = math.ceil(dataset.document_count / self.args.eval_batch_size)
        for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
            # move batch to selected device
            batch = util.to_device(batch, self._device)

            # run model (forward pass)
            result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                           entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                           entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                           dephead=batch['dephead'], deplabel=batch['deplabel'], pos=batch['pos'], evaluate=True)
            entity_clf, rel_clf, rels, pooler_output = result

            # evaluate batch
            pooler_output_all = torch.cat([pooler_output_all, pooler_output], dim=0)
            torch.save(pooler_output_all[1:, :], 'your/root/path/pooler_output' + Names + '.pt')  # Set your path here

            if rel_clf.shape[1] != 1:
                s = -1 * rel_clf * torch.log(rel_clf + 1e-12)
                s = torch.sum(s, dim=2, keepdim=False)
                a = sorted(s[0, :], key=abs, reverse=True)
                s = a[0].cpu().numpy()
            else:
                s = 0

            if entity_clf.shape[1] != 1:
                e = -1 * entity_clf * torch.log(entity_clf + 1e-12)
                e = torch.sum(e, dim=2, keepdim=False)
                size = e.shape[1]
                e = e.sum().item()
            else:
                e = 0

            prediction_entropy_entities = np.vstack((prediction_entropy_entities, e))
            torch.save(prediction_entropy_entities[1:, :], 'your/root/path/entropy_entities' + Names + '.pt')  # Set your path here

            sentence_size = np.vstack((sentence_size, size))
            torch.save(sentence_size[1:, :], 'your/root/path/sents_size' + Names + '.pt')  # Set your path here

            prediction_entropy_relation = np.vstack((prediction_entropy_relation, s))
            torch.save(prediction_entropy_relation[1:, :], 'your/root/path/entropy_relation' + Names + '.pt')  # Set your path here

            one = torch.ones_like(rel_clf)
            zero = torch.zeros_like(rel_clf)
            prediction_label = torch.where(rel_clf >= 0.4, one, zero)
            prediction_label_all = np.vstack((prediction_label_all, torch.sum(prediction_label, dim=1, keepdim=False)[0, :].cpu().numpy()))

            torch.save(prediction_label_all[1:, :], 'your/root/path/labelprediction' + Names + '.pt')  # Set your path here

            evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

    global_iteration = epoch * updates_epoch + iteration
    ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
    self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                   epoch, iteration, global_iteration, dataset.label)

    if self.args.store_predictions and not self.args.no_overlapping:
        evaluator.store_predictions()

    if self.args.store_examples:
        evaluator.store_examples()

    return ner_eval[2], rel_eval[2]


    def _get_optimizer_params(self, model):
        # param_optimizer = list(model.named_parameters())
        param_optimizer = filter(lambda p: p[1].requires_grad, model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
