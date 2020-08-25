import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer

from typing import List, Dict, Tuple


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained(
                "bert-large-uncased", cache_dir=temp_dir
            )
            # bert-uncased = 소문자
        else:
            self.model = BertModel.from_pretrained(
                "bert-base-uncased", cache_dir=temp_dir
            )
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec
        # top_vec = last hidden layer


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size,
            args.ext_ff_size,
            args.ext_heads,
            args.ext_dropout,
            args.ext_layers,
        )
        if args.encoder == "baseline":
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.ext_hidden_size,
                num_hidden_layers=args.ext_layers,
                num_attention_heads=args.ext_heads,
                intermediate_size=args.ext_ff_size,
            )
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size
            )
            my_pos_embeddings.weight.data[
                :512
            ] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[
                512:
            ] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                None, :
            ].repeat(
                args.max_pos - 512, 1
            )
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint["model"], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls)-> :
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
