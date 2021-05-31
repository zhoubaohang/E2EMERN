import os
import sys
import random
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
import tensorflow as tf
from model import E2EMERN
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--epoch", default=16, type=int)
parser.add_argument("--LAMBDA", default=1, type=float)
parser.add_argument("--MU", default=1, type=float)
parser.add_argument("--bert_path", default="./biobert_large", type=str)
parser.add_argument("--dataset", choices=["ncbi", "cdr"], default="ncbi", type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# dataset path
DICT_DATASET = DATASET[args.dataset]["to"]
dataLoader = DataLoader(args.bert_path, DICT_DATASET)

LABEL_SIZE = dataLoader.LABEL_SIZE

model = E2EMERN(args.bert_path, LABEL_SIZE)
optimizer = tf.keras.optimizers.Adam(1e-5)

count = 0
losses = []
dec_cnt = 0
ner_result = 0
best_epoch = 0
best_results = {}


def train_model(model, optimizer, data, count):
    for (
        train_s_ind,
        train_s_seg,
        train_e_ind,
        train_e_seg,
        train_ner,
        train_cpt_ner,
        train_nen,
        _,
    ) in tqdm(data, ascii=True):
        count += 1
        loss = train_one_step(
            model,
            optimizer,
            train_s_ind,
            train_s_seg,
            train_e_ind,
            train_e_seg,
            one_hot(train_ner, LABEL_SIZE),
            one_hot(train_cpt_ner, LABEL_SIZE),
            one_hot(train_nen, 2),
            args.MU,
            args.LAMBDA,
        )
        losses.append(loss)
    return count


for e in range(args.epoch):

    count = train_model(model, optimizer, dataLoader.Data("train"), count)
    dataLoader.resampling_data("train")

    devel_result = evaluate(model, dataLoader, "devel")
    devel_ner_f1 = "%.4f" % devel_result["ner"][-1]
    devel_nen_f1 = "%.4f" % devel_result["nen"][-1]

    step = "%5d" % count
    if ner_result < devel_result["ner"][-1]:
        best_results = devel_result.copy()
        ner_result = best_results["ner"][-1]
        best_epoch = e + 1
        dec_cnt = 0
    else:
        dec_cnt += 1
    loss = np.mean(losses)
    loss = "%4.6f" % loss
    losses.clear()

    description = (
        f"Step:{step} | Train loss: {loss} | Devel:{devel_ner_f1} {devel_nen_f1}"
    )
    file_log(f"{args.dataset}_result.txt", description)

    if dec_cnt == 3:
        best_epoch = best_epoch + 2
        break

description = f"Hyper-parameters: epoch:{best_epoch} lambda:{args.LAMBDA} mu:{args.MU}\nBest result:{best_results['ner']} {best_results['nen']}\n"
file_log(f"{args.dataset}_result.txt", description)
sys.stdout.write(f"{best_epoch} {args.LAMBDA} {args.MU}")
