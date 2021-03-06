import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from analysis import rocstories as rocstories_analysis
from analysis import pw as pw_analysis
from analysis import pw_retrieved as pw_retrieved_analysis
from datasets import pw, rocstories
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model, freeze_transformer_params
from opt import OpenAIAdam
from text_utils import TextEncoder, TextHideWordsEncoder, TextSelectIndexEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import MultipleChoiceLossCompute, ClassificationLossCompute

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
        x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

def combine_locs(x1, x2, loc):
    #combine context and part locs. add to compensate for premise and delimiters
    #save four locations for whole, two locations for part and jj, for size consistency
    new_loc = []

    for ix, (l1,l2) in enumerate(zip(*loc)):
        #spaghetti code to deal with multi-word wholes......
        if ix == 0:
            if type(l1) is tuple:
                #multi-word whole in sentence 1
                new_loc.extend([l1[0]+1, l1[1]+1])
            elif l1 != -1:
                #whole in sentence 1
                new_loc.extend([l1+1, l1+1])
            else:
                #whole must be in sentence 2
                if type(l2) is tuple:
                    new_loc.extend([l2[0]+len(x1)+2, l2[1]+len(x1)+2])
                else:
                    new_loc.extend([l2+len(x1)+2, l2+len(x1)+2])
            if type(l2) is tuple:
                #multi-word whole in sentence 2
                new_loc.extend([l2[0]+len(x1)+2, l2[1]+len(x1)+2])
            elif l2 != -1:
                #whole in sentence 2
                new_loc.extend([l2+len(x1)+2, l2+len(x1)+2])
            else:
                #whole must be in sentence 1
                if type(l1) is tuple:
                    new_loc.extend([l1[0]+1, l1[1]+1])
                else:
                    new_loc.extend([l1+1, l1+1])
        else:
            if l1 == -1:
                new_loc.extend([l2+len(x1)+2, l2+len(x1)+2])
            elif l2 == -1:
                new_loc.extend([l1+1, l1+1])
            else:
                new_loc.extend([l1+1, l2+len(x1)+2])
    return new_loc

def transform_pw(X1, X2, locs=None, hide_words=False):
    """
        Glue stories together with delimiter and stuff, and add position tokens
    """
    xmb = []#np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = []#np.zeros((n_batch, n_ctx), dtype=np.float32)
    #size is 8 - see combine_locs
    lmb = []#np.zeros((n_batch, 8), dtype=np.int32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    fields = (X1, X2, *locs) if locs else (X1, X2)

    #position tokens
    start = n_vocab
    end = n_vocab + n_ctx
    pos_toks = np.arange(start, end)

    idxs = []
    for idx,data in enumerate(zip(*fields)):
        if locs:
            x1, x2, *loc = data
            if len(x1) > max_len or len(x2) > max_len or len(x1) + len(x2) > n_ctx:
                #too long, don't deal
                continue
            loc = combine_locs(x1, x2, loc)
            if not any(loc) or not any(x1) or not any(x2):
                continue
            lmb.append(loc)
        else:
            x1, x2 = data
        #concatenate
        x = [start] + x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        l = len(x)
        #set np array
        #xmb[i,:l,0] = x
        xmb.append(np.vstack((x + [0] * (n_ctx - l), pos_toks)))
        #mask
        #mmb[i,:l] = 1
        mmb.append([1] * l + [0] * (n_ctx - l))
        idxs.append(idx)
    return np.array(xmb), np.array(mmb), np.array(lmb), idxs

def iter_apply(Xs, Ms, Ys, Ls=None):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    fields = (Xs, Ms, Ys, Ls) if Ls is not None else (Xs, Ms, Ys)
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for field in iter_data(*fields, n_batch=n_batch_train, truncate=False, verbose=True):
            if len(fields) == 3:
                xmb, mmb, ymb = field
            else:
                xmb, mmb, ymb, lmb = field
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            if len(fields) > 3:
                LMB = torch.tensor(lmb, dtype=torch.long).to(device)
                _, clf_logits = dh_model(XMB, LMB)
            else:
                _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, Ms, Ls=None):
    logits = []
    fields = (Xs, Ms, Ls) if Ls is not None else (Xs, Ms)
    with torch.no_grad():
        dh_model.eval()
        for field in iter_data(*fields, n_batch=n_batch_train, truncate=False, verbose=True):
            if len(fields) == 2:
                xmb, mmb = field
            else:
                xmb, mmb, lmb = field
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            if len(fields) > 2:
                LMB = torch.tensor(lmb, dtype=torch.long).to(device)
                _, clf_logits = dh_model(XMB, LMB)
            else:
                _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_inps = trX[:n_valid], trM[:n_valid], trY[:n_valid]
    va_inps = vaX, vaM, vaY
    if hard_select:
        tr_inps = (*tr_inps, trL[:n_valid])
        va_inps = (*va_inps, vaL)
    tr_logits, tr_cost = iter_apply(*tr_inps)
    va_logits, va_cost = iter_apply(*va_inps)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc + '_' + exec_time, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def predict(dataset, submission_dir, test):
    filename = filenames[dataset].replace('.tsv', f'_{exec_time}.tsv')
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    if test:
        fields = (teX, teM, teL) if hard_select else (teX, teM)
    else:
        fields = (vaX, vaM, vaL) if hard_select else (vaX, vaM)
    predictions = pred_fn(iter_predict(*fields))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))
    return path


def run_epoch(fields):
    for ix,field in enumerate(iter_data(*shuffle(*fields, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True)):
        global n_updates
        if n_gpu > 1 and np.isnan(dh_model.module.transformer.embed.weight.data.cpu().numpy()).any():
            print("null")
            import pdb; pdb.set_trace()
        elif n_gpu <= 1 and np.isnan(dh_model.transformer.embed.weight.data.cpu().numpy()).any():
            print("null")
            import pdb; pdb.set_trace()
        dh_model.train()
        if len(fields) == 3:
            xmb, mmb, ymb = field
        else:
            xmb, mmb, ymb, lmb = field
        XMB = torch.tensor(xmb, dtype=torch.long).to(device).transpose(1,2)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb, dtype=torch.float).to(device)
        if len(fields) > 3:
            LMB = torch.tensor(lmb, dtype=torch.long).to(device)
            lm_logits, clf_logits = dh_model(XMB, LMB)
        else:
            lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)#, debug=ix==39)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
    'pw': argmax,
    'pw-retrieved': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
    'pw': 'pw_preds.tsv',
    'pw-retrieved': 'pw_retrieved_preds.tsv',
}

label_decoders = {
    'rocstories': None,
    'pw': None,
    'pw-retrieved': None,
}

analyses = {
    'rocstories': rocstories_analysis,
    'pw': pw_analysis,
    'pw-retrieved': pw_retrieved_analysis,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', choices=['pw', 'rocstories', 'pw-retrieved'])
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--params_path',    type=str, default='model/params_shapes.json', help="path to params file if using previously finetuned params")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--ordinal', action='store_true', help="flag to do 5-class prediction instead of binary")
    parser.add_argument('--test', action='store_true', help="flag to run on test")
    parser.add_argument('--freeze_lm', action='store_true', help="flag to freeze (not update) LM weights - only train the classifier")
    parser.add_argument('--hard_select', action='store_true', help="flag to use as final layer representation the concatenation of hidden states at appropriate indices of input")
    parser.add_argument('--hide_words', action='store_true', help="flag to replace whole, part, and jj words with special tokens")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    args.exec_time = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir
    hard_select = args.hard_select
    exec_time = args.exec_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    log_file = os.path.join(log_dir, '{}_{}.jsonl'.format(desc, exec_time))
    logger = ResultLogger(path=log_file, **args.__dict__)
    # formatting stuff
    if hard_select:
        text_encoder = TextSelectIndexEncoder(args.encoder_path, args.bpe_path)
    elif args.hide_words:
        text_encoder = TextHideWordsEncoder(args.encoder_path, args.bpe_path)
    else:
        text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print("Encoding dataset...")
    if args.dataset == 'rocstories':
        (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(rocstories(data_dir), encoder=text_encoder)
    elif args.dataset.startswith('pw'):
        train_file = os.path.join(data_dir, ('retrieved_2_train_feats.jsonl') if args.dataset == 'pw-retrieved' else 'snli_style_train_feats.jsonl')
        if hard_select or args.hide_words:
            trdata, dvdata, tedata, triples = pw(train_file, args.ordinal, True)
            if hard_select:
                texts_tokens, locs = encode_dataset((trdata, dvdata, tedata), encoder=text_encoder, triples=triples)
                (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2, teY) = texts_tokens
            else:
                (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2, teY) = encode_dataset((trdata, dvdata, tedata), encoder=text_encoder, triples=triples)
        else:
            (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2, teY) = encode_dataset(pw(train_file, args.ordinal, hard_select), encoder=text_encoder)
    #output: unpadded lists of word indices

    #special token
    clf_token = encoder['_classify_']

    #number of special characters
    n_special = 6 if args.hide_words else 3

    max_len = n_ctx // 2 - 2
    n_additions = 0
    #get max length of story + answer in train, val, test
    #take min of (that + 3), n_ctx
    #the 3 is to take care of the special start, delimiter, end tokens
    if args.dataset == 'rocstories':
        n_ctx = min(max([len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)])+n_special, n_ctx)
    elif args.dataset.startswith('pw'):
        n_ctx = min(max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(trX1, trX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(vaX1, vaX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(teX1, teX2)])+n_special, n_ctx)

    if args.dataset == 'rocstories':
        trX, trM = transform_roc(trX1, trX2, trX3)
        vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
        if submit:
            teX, teM = transform_roc(teX1, teX2, teX3)
    elif args.dataset.startswith('pw'):
        if hard_select:
            print("encoding locs")
            trX, trM, trL, idxs = transform_pw(trX1, trX2, locs[0], hide_words=args.hide_words)
            trY = trY[idxs]
            vaX, vaM, vaL, idxs = transform_pw(vaX1, vaX2, locs[1], hide_words=args.hide_words)
            vaY = vaY[idxs]
            if submit:
                teX, teM, teL, idxs = transform_pw(teX1, teX2, locs[2], hide_words=args.hide_words)
                teY = teY[idxs]
        else:
            trX, trM, _, idxs = transform_pw(trX1, trX2, hide_words=args.hide_words)
            trY = trY[idxs]
            vaX, vaM, _, idxs = transform_pw(vaX1, vaX2, hide_words=args.hide_words)
            vaY = vaY[idxs]
            if submit:
                teX, teM, _, idxs = transform_pw(teX1, teX2, hide_words=args.hide_words)
                teY = teY[idxs]

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    if args.dataset == 'rocstories':
        task = 'multiple_choice'
    elif args.dataset.startswith('pw'):
        task = ('inference', 5 if args.ordinal else 2)
    #if hide words, ignore the extra three we added
    vocab = n_vocab + n_ctx
    dh_model = DoubleHeadModel(args, clf_token, task, vocab, n_ctx, hard_select=hard_select)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    if args.dataset == 'rocstories':
        compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                     criterion,
                                                     args.lm_coef,
                                                     model_opt)
    elif args.dataset.startswith('pw'):
        compute_loss_fct = ClassificationLossCompute(criterion,
                                                     criterion,
                                                     args.lm_coef,
                                                     model_opt)

    if args.params_path == 'model/params_shapes.json':
        load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
    else:
        sd = torch.load(args.params_path)
        sd = {(name[7:] if name.startswith('module.') else name):val for name,val in sd.items()}
        dh_model.load_state_dict(sd)
        args.n_iter = 0

    if args.freeze_lm:
        print("freezing params")
        freeze_transformer_params(dh_model)

    dh_model.to(device)
    if n_gpu > 1:
        dh_model = nn.DataParallel(dh_model)

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    #if submit:
    #    path = os.path.join(save_dir, desc, 'best_params')
    #    torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    fields = (trX, trM, trYt, trL) if hard_select else (trX, trM, trYt)
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch(fields)
        n_epochs += 1
        log(save_dir, desc)

    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        dh_model.load_state_dict(torch.load(path))
        pred_path = predict(dataset, args.submission_dir, args.test)
        if args.analysis:
            analy_fn = analyses[dataset]
            analy_fn(data_dir, pred_path, os.path.join(log_dir, log_file), test=args.test, ordinal=args.ordinal)
