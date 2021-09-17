# -*- coding: utf-8 -*-
# AUTHOR: Hang Yang
# DATE: 21-7-11

import logging
import os
import re
from collections import defaultdict, Counter
import numpy as np
import torch
from .dee_metric import measure_event_table_filling
from .event_type import event_type2event_class, BaseEvent, event_type_fields_list, common_fields
from .ner_task import NERExample, NERFeatureConverter
from .utils import default_load_json, default_dump_json, default_dump_pkl, default_load_pkl

logger = logging.getLogger(__name__)

class DEEFeature(object):
    def __init__(self, guid, ex_idx, doc_token_id_mat, doc_token_mask_mat, doc_token_label_mat,
                 span_token_ids_list, span_dranges_list, event_type_labels, event_arg_idxs_objs_list,
                 valid_sent_num=None):
        self.guid = guid
        self.ex_idx = ex_idx  # example row index, used for backtracking
        self.valid_sent_num = valid_sent_num

        # directly set tensor for dee feature to save memory
        # self.doc_token_id_mat = doc_token_id_mat
        # self.doc_token_mask_mat = doc_token_mask_mat
        # self.doc_token_label_mat = doc_token_label_mat
        self.doc_token_ids = torch.tensor(doc_token_id_mat, dtype=torch.long)
        self.doc_token_masks = torch.tensor(doc_token_mask_mat, dtype=torch.uint8)  # uint8 for mask
        self.doc_token_labels = torch.tensor(doc_token_label_mat, dtype=torch.long)

        # sorted by the first drange tuple
        # [(token_id, ...), ...]
        # span_idx -> span_token_id tuple
        self.span_token_ids_list = span_token_ids_list
        # [[(sent_idx, char_s, char_e), ...], ...]
        # span_idx -> [drange tuple, ...]
        self.span_dranges_list = span_dranges_list

        # [event_type_label, ...]
        # length = the total number of events to be considered
        # event_type_label \in {0, 1}, 0: no 1: yes
        self.event_type_labels = event_type_labels
        # event_type is denoted by the index of event_type_labels
        # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        # if no event objects, event_type_idx -> None
        self.event_arg_idxs_objs_list = event_arg_idxs_objs_list

        # event_type_idx -> event_field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        self.event_idx2field_idx2pre_path2cur_span_idx_set = self.build_dag_info(self.event_arg_idxs_objs_list)

        # event_type_idx -> key_sent_idx_set, used for key-event sentence detection
        self.event_idx2key_sent_idx_set, self.doc_sent_labels = self.build_key_event_sent_info()


    def generate_dag_info_for(self, pred_span_token_tup_list, return_miss=False):
        '''
        :param pred_span_token_tup_list:  entity span token id (pred or gold)
        '''
        num_pred_span = len(pred_span_token_tup_list)
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
            else:
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        pred_event_arg_idxs_objs_list = []
        pred_event_type_idxs_list = []
        # one_event_type = False
        for i, (event_arg_idxs_objs, event_type_idxs) in enumerate(zip(self.event_arg_idxs_objs_list, self.event_type_labels)):
            if event_arg_idxs_objs is None:
                continue
            else:
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                gold_span_idx2pred_span_idx[gold_span_idx]
                            )
                        else:
                            pred_event_arg_idxs.append(num_pred_span)
                    pred_event_type_idxs_list.append(i)
                    pred_event_arg_idxs_objs_list.append(tuple(pred_event_arg_idxs))
        return gold_span_idx2pred_span_idx, pred_event_arg_idxs_objs_list, pred_event_type_idxs_list

    def get_event_args_objs_list(self):
        event_args_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                event_args_objs_list.append(None)
            else:
                event_args_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    event_args = []
                    for arg_idx in event_arg_idxs:
                        if arg_idx is None:
                            token_tup = None
                        else:
                            token_tup = self.span_token_ids_list[arg_idx]
                        event_args.append(token_tup)
                    event_args_objs.append(event_args)
                event_args_objs_list.append(event_args_objs)

        return event_args_objs_list




