import pickle
from typing import Any

import pytorch_lightning as pl
import torch
from transformers import AutoModel
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from pytorch_lightning.loggers import TensorBoardLogger

from data_pl import DataPL, cls_to_int


class ClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, do, n_labels):
        super().__init__()
        self.n_labels = n_labels
        self.dense = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.6)
        self.do = nn.Dropout(do)
        self.output = nn.Linear(hidden_size, n_labels)

    def forward(self, batch):
        # batch.shape = [batch_size, max_seq_len, hidden_dim]
        batch = self.do(batch)
        batch = self.dense(batch)

        batch = self.bn(batch.permute(0,2,1))
        batch = F.tanh(batch.permute(0,2,1))

        batch = self.do(batch)
        output = self.output(batch)

        return output


class SeqLabeling(pl.LightningModule):
    def __init__(self, pretrained_model, head, lr, softmax_temp, lr_decay, fine_tuning=False):
        super().__init__()
        self.model = pretrained_model
        self.softmax_temp = softmax_temp
        self.head = head
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_classes = head.n_labels
        self.fine_tuning = fine_tuning

    def forward(self, batch, mask):
        outputs = self.model(batch, attention_mask=mask)
        # named_tuple with
        #   `last_hidden_state` of shape torch.Size([batch_size, max_seq_len, hid_dim])
        #   `pooler_output` of shape torch.Size([batch_size, hid_dim])

        last_hidden = outputs.last_hidden_state
        outputs = self.head(last_hidden)
        # outputs.shape: torch.Size([batch_size, max_seq_len, n_classes])

        return outputs

    def _step(self, batch):
        inp, trg, mask, seq_lens = batch['inp'], batch['trg'], batch['msk'], batch['seq_lens']
        predictions = self(inp, mask)
        # if self.training:
        predictions /= self.softmax_temp

        # predictions.shape = [batch_size, max_seq_len, n_classes]

        loss = 0
        acc = 0
        for seq, seq_len, t in zip(predictions, seq_lens, trg):
            loss += F.cross_entropy(seq[:seq_len], t[:seq_len], reduction='sum')
            predicted = seq.argmax(1)
            acc += torch.sum(predicted == t)
        # ic(loss, acc, sum(seq_lens))

        return loss / sum(seq_lens), acc / sum(seq_lens), sum(seq_lens)

    def training_step(self, batch, idx):
        loss, acc, batch_size = self._step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def validation_step(self, batch, idx):
        loss, acc, batch_size = self._step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def test_step(self, batch, idx):
        loss, acc, batch_size = self._step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inp, trg, mask, seq_lens = batch['inp'], batch['trg'], batch['msk'], batch['seq_lens']
        outputs = self(inp, mask) / self.softmax_temp
        predicted = []
        targets = []

        for seq, seq_len, t in zip(outputs, seq_lens, trg):
            new_seq, new_t = seq[:seq_len], t[:seq_len]
            predicted.append(new_seq.argmax(1).detach().cpu().numpy())
            targets.append(new_t.cpu().numpy())

        return dict(
            predicted=predicted,
            trg=targets,
            batch=batch
        )

    def configure_optimizers(self):
        print('::::::::::::::::::::::::::: configure optimizers :::::::::::::::::::::::::::')
        if not self.fine_tuning:
            optimizer = optim.Adam(self.head.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def _aggregate_output(self, outputs):
        loss, acc, total = 0, 0, 0
        for x in outputs:
            total += x['batch_size']
            loss += x['loss'] * x['batch_size']
            acc += x['acc'] * x['batch_size']
        return dict(loss=loss, acc=acc, cnt=total)

    def _epoch_end(self, outputs, name):
        # name can be {Train, Test, Val}
        epoch_output = self._aggregate_output(outputs)

        # now need to get aggregated output from all gpus into one place
        # done with all_gather, ... see https://github.com/PyTorchLightning/pytorch-lightning/discussions/6501#discussioncomment-589529
        # before all_gather aggregated output was a dict of numbers (floats and ints)
        # after all_gather it will be a dict of lists, and list length will be equal to the world size
        # ic(epoch_output)
        # ic(outputs)
        # print(name, epoch_output)
        if torch.cuda.device_count() > 1:
            epoch_output = self.all_gather(epoch_output)
            epoch_output = {k: sum(val) for k, val in epoch_output.items()}
        acc = epoch_output['acc'] / epoch_output['cnt']
        loss = epoch_output['loss'] / epoch_output['cnt']

        # log only at rank 0
        if self.global_rank == 0:
            self.logger.experiment.add_scalar(f'Loss/{name}', loss, self.current_epoch)
            self.logger.experiment.add_scalar(f'Acc/{name}', acc, self.current_epoch)
        return acc, loss

    def validation_epoch_end(self, outputs):
        val_acc, val_loss = self._epoch_end(outputs, 'Val')
        self.log('val_acc', val_acc, logger=False, sync_dist=True)
        self.log('val_loss', val_loss, logger=False, sync_dist=True)

    def training_epoch_end(self, outputs):
        _, _ = self._epoch_end(outputs, 'Train')

    def test_epoch_end(self, outputs):
        _, _ = self._epoch_end(outputs, 'Test')

    def set_fine_tune(self, val):
        self.fine_tuning = val

    def set_lr(self, lr):
        self.lr = lr

    def set_lr_decay(self, decay):
        self.lr_decay = decay


if __name__ == '__main__':
    pl_data = DataPL(
        data_fn='data_aligned.pickle',
        orig_fn='data.json',
        train_fn='train_indices.pickle',
        dev_fn='val_indices.pickle',
        test_fn='test_indices.pickle',
        tokenizer_name='ufal/robeczech-base',
        batch_size=256,
        n_workers=10,
    )

    pl_model = SeqLabeling(
        pretrained_model=AutoModel.from_pretrained('ufal/robeczech-base'),
        head=ClassificationHead(
            input_size=768,
            hidden_size=256,
            do=0.5,
            n_labels=len(cls_to_int)
        ),
        lr=2e-4,
        # epochs=2,
        lr_decay=0.95,
        softmax_temp=0.2,
    )

    logger = TensorBoardLogger('inv_norm')

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        # strategy='ddp',
        logger=logger,
        # callbacks=cb,
        num_sanity_val_steps=0,
        deterministic=True,
        precision=16,
        enable_checkpointing=False,
        max_epochs=15,
        # limit_train_batches=2,
        # limit_val_batches=2,
        # fast_dev_run=2,
    )

    trainer.fit(pl_model, pl_data)

    print('first train done')

    trainer.fit_loop.max_epochs *= 4
    pl_model.set_fine_tune(True)
    pl_model.set_lr(1e-5)
    pl_model.set_lr_decay(0.92)

    trainer.fit(pl_model, pl_data)

    trainer.test(pl_model, pl_data)

    predictions = trainer.predict(pl_model, pl_data.test_dataloader())
    with open('predictions.pickle', 'wb') as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
