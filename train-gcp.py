# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from google.cloud import storage

parser = argparse.ArgumentParser(description='Comment fine-tuned KoGPT2')

parser.add_argument('--generate',
                    action='store_true',
                    default=False)

parser.add_argument('--inputs',
                    type=str,
                    default='진짜')

parser.add_argument('--model_params',
                    type=str,
                    default='NONE')

parser.add_argument('--dataset_path',
                    type=str,
                    default='NONE')

parser.add_argument('--train',
                    action='store_true',
                    default=False)

parser.add_argument('--bucket',
                    type=str,
                    default='NONE')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BOS = '<s>'
EOS = '</s>'
MASK = '<mask>'
NEWLINE = '<unused0>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

class CommentDataset(Dataset):
    def __init__(self, comments, max_len=32):
        self._data = comments
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER

        temp = []

        for x in self._data:
            toked = self.tokenizer.tokenize(x)

            while len(toked) > self.max_len - 2:
                temp.append([self.bos] + toked[:self.max_len - 2] + [self.eos])
                toked = toked[self.max_len - 1:]
            
            temp.append([self.bos] + toked + [self.eos])

        self._data = temp

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        toked = self._data[idx]
        
        token_ids = self.tokenizer.convert_tokens_to_ids(toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
            
        return token_ids

class CommentKoGPT2(LightningModule):
    def __init__(self, hparams, bucket='', dataset_path='', **kwargs) -> None:
        super(CommentKoGPT2, self).__init__()
        self.save_hyperparameters(hparams)
        self.dataset_path = dataset_path
        self.bucket = bucket
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs, labels):
        output = self.kogpt2(inputs, labels=labels, return_dict=True)
        
        return output

    def training_step(self, batch, batch_idx):
        token_ids = batch
        outputs = self.forward(token_ids, token_ids)
        
        loss, logits = outputs[:2]
        self.log('train_log', loss)
        
        return loss

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        return torch.LongTensor(batch)

    def train_dataloader(self):
        if self.bucket != '':
            client = storage.Client()
            bucket = client.get_bucket('pytorchtpu')
            blob = bucket.get_blob(self.dataset_path)
            data = blob.download_as_string().splitlines()
        else:
            with open(self.dataset_path, 'r') as f:
                data = f.readlines()
        self.train_set = CommentDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=1,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def generate(self, inputs):
        tokenizer = TOKENIZER
        input_ids = tokenizer.encode(inputs, add_special_tokens=False, return_tensors="pt")
        outputs = model.kogpt2.generate(input_ids=input_ids, do_sample=True, max_length=32, repetition_penalty=2.0, top_k=50, top_p=0.95, num_return_sequences=3)

        for index, i in enumerate(outputs):
            print("{0}: {1}".format(index + 1, tokenizer.decode(i, skip_special_tokens=True)))

parser = CommentKoGPT2.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)
      
if __name__ == "__main__":
    if args.train:
        # python train_torch.py --train --gpus 1 --max_epochs 3
        checkpoint_callback = ModelCheckpoint(
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            mode='min',
            prefix='model_'
        )
        if args.model_params == "NONE":
            model = CommentKoGPT2(args, dataset_path=args.dataset_path)
        else:
            model = CommentKoGPT2.load_from_checkpoint(args.model_params, dataset_path=args.dataset_path)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    elif args.generate:
        model = CommentKoGPT2.load_from_checkpoint(args.model_params)
        model.generate(args.inputs)