import os
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
parser.add_argument("--save_weights", action="store_true")
parser.add_argument("--save_pred_result", action="store_true")
parser.add_argument("--weights", default="./weights", type=str)
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
best_result = {}
min_loss = np.inf


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
    count = train_model(model, optimizer, dataLoader.Data("devel"), count)

    devel_result = evaluate(model, dataLoader, "devel")
    devel_ner_f1 = "%.4f" % devel_result["ner"][-1]
    devel_nen_f1 = "%.4f" % devel_result["nen"][-1]
    dataLoader.resampling_data("devel")

    test_result = evaluate(model, dataLoader, "test")
    test_ner_f1 = "%.4f" % test_result["ner"][-1]
    test_nen_f1 = "%.4f" % test_result["nen"][-1]
    if args.save_pred_result:
        save_prediction_result(model, dataLoader, f"results/step{count}.tsv", "test")

    zs_test_result = evaluate(model, dataLoader, "zs_test")
    zs_test_ner_f1 = "%.4f" % zs_test_result["ner"][-1]
    if args.save_pred_result:
        save_prediction_result(
            model, dataLoader, f"results/zs_step{count}.tsv", "zs_test"
        )

    step = "%5d" % count
    loss = np.mean(losses)
    if loss < min_loss:
        min_loss = loss
        best_result = {"test": test_result.copy(), "zs_test": zs_test_result.copy()}
        if args.save_weights:
            model.save_weights(f"{args.weights}/{args.dataset}_model.h5")
    loss = "%4.6f" % loss
    losses.clear()

    description = f"Step:{step} | Train loss: {loss} | Devel:{devel_ner_f1} {devel_nen_f1} | Test:{test_ner_f1} {test_nen_f1} | Zero-shot Test:{zs_test_ner_f1}"
    tqdm.write(description)
    file_log(f"{args.dataset}_result.txt", description)

test_result = best_result["test"]
zs_test_result = best_result["zs_test"]
description = f"Test:{test_result['ner'][-1]} {test_result['nen'][-1]} | Zero-shot Test:{zs_test_result['ner'][-1]}\n"
print(description)
file_log(f"{args.dataset}_result.txt", description)
