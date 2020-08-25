import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from others.tokenization import BertTokenizer

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos   = '[unused0]'
        self.tgt_eos   = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        # sep / cls / pad token id

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test = False):
        # src --> src sentences / src = [sent1, sent2, ...]
        # sent1 = [word1, word2, word3, ...]
        
        
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idx = [i for i,s in enumerate(src) if(len(s) > self.args.min_src_ntokens_per_sent)]
        # src의 word / token의 갯수가 특정 min 갯수 이상이여야 idx에 포함
        # idx = src sentence들의 index
        # args.min_src_ntokens_per_sent


        _sent_labels = [0] * len(src)
        # 문장 갯수 만큼 [0] *len(src) = [0, 0, 0, 0 ...]
        for l in sent_labels:
            _sent_labes[l] = 1

        # ' {} {} '.format('a', 'b').join(['1', '2', '3'])

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        # idx에 있는 index의 src sentence의 max token만큼 짤라버린다.
        # args.max_src_ntokens_per_sent
        # 여기까지 src는 sentence들이 담겨있다 (Yet, Not Tokenized)

        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        # sent_labels에는 문장마다의 label이 존재하고
        # max_src_nsents: src 문장 최대 갯수 만큼 자른다.

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        # ' {} {} '.format('[SEP]', '[CLS]').join(['I Love Professor', 'And He Likes me too', 'I Love studying'])
        # text = 'I Love Professor [SEP] [CLS] And He Likes me too [SEP] [CLS] I Love studying'

        src_subtokens = self.tokenizer.tokenize(text)
        # tokenizer가 wordpiece라 하나의 token도 여러 token으로 분리될 수 있기에,
        # subtokens라 부른다.
        # tokenize에 .split()가 있어, 그냥 raw text 단으로 넣어도 된다.

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        # [CLS]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i,t in enumerate(src_subtokens_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        # _segs는 [SEP]의 index [-1, 0, 0, 0 ...]
        # segs는 두 [SEP]의 간격
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i,t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        )

        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt



