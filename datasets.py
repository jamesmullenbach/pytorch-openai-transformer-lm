import os
import csv
import json
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def read_pws(path, ordinal, hard_select):
    with open(path) as f:
        prems = []
        hyps = []
        y = []
        trips = []
        for line in f:
            example = json.loads(line.strip())
            prems.append(example['sentence1'])
            hyps.append(example['sentence2'])
            if hard_select:
                trips.append((example['whole'], example['part'], example['jj']))
            y.append(int(example['label' if ordinal else 'bin_label']))
        return prems, hyps, y, trips

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

def pw(data_dir, ordinal, hard_select):
    tr_prems, tr_hyps, trY, trtrips = read_pws(os.path.join(data_dir, 'snli_style_train_feats.jsonl'), ordinal, hard_select)
    dv_prems, dv_hyps, dvY, dvtrips = read_pws(os.path.join(data_dir, 'snli_style_dev_feats.jsonl'), ordinal, hard_select)
    te_prems, te_hyps, teY, tetrips = read_pws(os.path.join(data_dir, 'snli_style_test_feats.jsonl'), ordinal, hard_select)
    trY = np.asarray(trY, dtype=np.int32)
    dvY = np.asarray(dvY, dtype=np.int32)
    teY = np.asarray(teY, dtype=np.int32)
    if any(trtrips) and any(dvtrips) and any(tetrips):
        return (tr_prems, tr_hyps, trY), (dv_prems, dv_hyps, dvY), (te_prems, te_hyps, teY), (trtrips, dvtrips, tetrips)
    else:
        return (tr_prems, tr_hyps, trY), (dv_prems, dv_hyps, dvY), (te_prems, te_hyps, teY)
