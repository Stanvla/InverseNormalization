# %%
import pickle
import os
from data_pl import cls_to_int, int_to_cls
from collections import Counter

def align(text_to_id, preds):
    idx = 0
    result = []
    for w, ids in text_to_id:
        # print(w)
        tmp = []
        for id in ids:
            tmp.append(preds[idx])
            idx += 1
        tag = Counter(tmp).most_common(1)[0][0]
        if 'case' in tag:
            w = w[0].upper() + w[1:]
        if 'punc' in tag:
            punc = tag.split('_')[-1]
            w += punc
        elif tag == 'ok':
            pass
        else:
            if 'num' in tag:
                w = f'{w}[{tag}]'
        result.append(w)
    return ''.join(result[1:-1])
        # print(f'{w:25} [{tag}]')

# %%
if __name__ == '__main__':
    # %%
    # os.listdir()
    # os.getcwd()
    # %%
    with open('predictions.pickle', 'rb') as f:
        predictions = pickle.load(f)

    flat_predictions = dict(
        predicted=[],
        # trg=[],
        text_to_id=[],
        orig=[]
    )

    for batch in predictions:
        flat_predictions['predicted'].extend(batch['predicted'])
        # flat_predictions['trg'].extend(batch['trg'])
        flat_predictions['text_to_id'].extend(batch['batch']['text_to_id'])
        flat_predictions['orig'].extend(batch['batch']['orig'])

    final_predictions = []
    for text_to_id, orig, preds in zip(flat_predictions['text_to_id'], flat_predictions['orig'], flat_predictions['predicted']):
        # print(orig)
        # print(text_to_id)
        final_predictions.append([
            orig,
            align(text_to_id, [int_to_cls[p] for p in preds]),
        ])
        # print()
        print()
    # %%
    with open('result.txt', 'w') as f:
        for orig, pred in final_predictions:
            f.write(orig)
            f.write('\n')
            f.write(pred)
            f.write('\n')
            f.write('\n')
    # %%
    # %%
    # %%
    # %%
    # %%
    # %%
    # %%
    # %%
    # %%
    pass

