# %%
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from icecream import ic
import json

import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl


def pickle_read(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


cls_to_int = {
    "case_punc_!": 0,
    "case_punc_:": 1,
    "case_punc_?": 2,
    "punc_:": 3,
    "num_ord": 4,
    "punc_!": 5,
    "case_punc_.": 6,
    "case_punc_,": 7,
    "punc_?": 8,
    "num_card": 9,
    "punc_,": 10,
    "punc_.": 11,
    "case": 12,
    "ok": 13,
}

int_to_cls = {v: k for k, v in cls_to_int.items()}


class Data(Dataset):
    def __init__(self, data_fn, orig_data_fn, indices_fn, tokenizer):
        self.data = pickle_read(data_fn)
        
        with open(orig_data_fn, 'r') as f:
            self.orig_data = json.load(f)
        
        self.indices = pickle_read(indices_fn)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        idx = self.indices[idx]
        orig = self.orig_data[f'{idx}']['raw']
        norm, trg = self.data[idx]['norm'], self.data[idx]['trg']
        norm = [x if i == 0 else ' ' + x for i, x in enumerate(norm)]

        id_to_text = [[self.tokenizer.bos_token_id, self.tokenizer.bos_token]]
        text_to_id = [[self.tokenizer.bos_token, [self.tokenizer.bos_token_id]]]
        new_norm = [self.tokenizer.bos_token_id]
        new_trg = [0]

        for word, tag in zip(norm, trg):
            tokenized_word = self.tokenizer(word)['input_ids'][1:-1]
            word_trg = cls_to_int[tag]
            text_to_id.append([word, tokenized_word])
            for token in tokenized_word:
                id_to_text.append([token, word])
                new_trg.append(word_trg)

            new_norm.extend(tokenized_word)

        id_to_text.append([self.tokenizer.eos_token_id, self.tokenizer.eos_token])
        text_to_id.append([self.tokenizer.eos_token, [self.tokenizer.eos_token_id]])
        new_norm.append(self.tokenizer.eos_token_id)
        new_trg.append(13)

        # ic(text_to_id)
        # ic(len(id_to_text), id_to_text)
        # ic(len(new_norm), new_norm)
        # ic(len(new_trg), new_trg)

        # ic(idx, len(new_trg), len(new_norm), len(id_to_text), len(text_to_id))

        return dict(
            inp=new_norm, 
            trg=new_trg, 
            id_to_text=id_to_text, 
            text_to_id=text_to_id,
            orig=orig
        )

    def __len__(self):
        return len(self.indices)


class Collator:
    def __init__(self, pad_id=1, ok_id=0):
        self.pad_id = pad_id
        self.ok_id = ok_id

    def __call__(self, batch):
        xs = [x['inp'] for x in batch]
        ys = [x['trg'] for x in batch]

        max_len = max([len(x) for x in xs])
        seq_sizes = []
        inp_batch = []
        trg_batch = []
        msk_batch = []

        for x, y in zip(xs, ys):
            msk = torch.tensor([1] * len(x) + [0] * (max_len - len(x)))
            inp = torch.tensor(x + [self.pad_id] * (max_len - len(x)))
            # may be should be FloatTensor, idk
            trg = torch.tensor(y + [self.ok_id] * (max_len - len(x)))

            # ic(x, y)
            # ic(len(x), len(y))
            # ic(inp.shape, trg.shape, msk.shape)
            inp_batch.append(inp)
            trg_batch.append(trg)
            msk_batch.append(msk)
            seq_sizes.append(len(x))

        return dict(
            inp=torch.stack(inp_batch),
            trg=torch.stack(trg_batch),
            msk=torch.stack(msk_batch),
            text_to_id=[x['text_to_id'] for x in batch],
            id_to_text=[x['id_to_text'] for x in batch],
            orig=[x['orig'] for x in batch],
            seq_lens=seq_sizes,
        )


class DataPL(pl.LightningDataModule):
    def __init__(self, data_fn, orig_fn, train_fn, dev_fn, test_fn, tokenizer_name, batch_size, n_workers):
        super().__init__()
        self.bs = batch_size
        self.n_workers = n_workers
        self.pin_mem = torch.cuda.is_available()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_dataset = Data(
            data_fn=data_fn,
            orig_data_fn=orig_fn,
            indices_fn=train_fn,
            tokenizer=tokenizer
        )
        self.dev_dataset = Data(
            data_fn=data_fn,
            orig_data_fn=orig_fn,
            indices_fn=dev_fn,
            tokenizer=tokenizer
        )
        self.test_dataset = Data(
            data_fn=data_fn,
            orig_data_fn=orig_fn,
            indices_fn=test_fn,
            tokenizer=tokenizer,
        )

    def _get_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=Collator(),
            pin_memory=self.pin_mem,
            num_workers=self.n_workers,
        )

    def train_dataloader(self):
        return self._get_loader(self.train_dataset)

    def val_dataloader(self):
        return self._get_loader(self.dev_dataset)

    def test_dataloader(self):
        return self._get_loader(self.test_dataset)


# %%

if __name__ == '__main__':
    # %%
    params = dict(
        data_fn='data_aligned.pickle',
        orig_fn='data.json',
        train_fn='train_indices.pickle',
        dev_fn='val_indices.pickle',
        test_fn='test_indices.pickle',
        tokenizer_name='ufal/robeczech-base',
        batch_size=2,
        n_workers=0,
    )
    # res = pickle_read(data)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # %%
    # train_dataset = Data(
    #     data_fn=data,
    #     orig_data_fn='data.json',
    #     indices_fn='train_indices.pickle',
    #     tokenizer=tokenizer,
    # )
    # for i in range(len(train_dataset)):
    #     train_dataset[i]
    # %%
    data_pl = DataPL(**params)
    # test_loader = data_pl.test_dataloader()
    for b in data_pl.test_dataloader():
        pass

    for b in data_pl.val_dataloader():
        pass

    for b in data_pl.train_dataloader():
        pass

    # %%
    # %%
    # %%
    # %%
    # %%
    # %%


