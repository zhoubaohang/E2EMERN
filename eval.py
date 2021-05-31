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
parser.add_argument("--LAMBDA", default=1, type=float)
parser.add_argument("--MU", default=1, type=float)
parser.add_argument("--save_pred_result", action="store_true")
parser.add_argument("--weights", default="./weights", type=str)
parser.add_argument("--bert_path", default="./biobert_large", type=str)
parser.add_argument("--dataset", choices=["ncbi", "cdr"], default="ncbi", type=str)
args = parser.parse_args()

if not os.path.exists(args.weights):
    print(f"Could not find the weights file {args.weights}")
    exit()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# dataset path
DICT_DATASET = DATASET[args.dataset]["to"]
dataLoader = DataLoader(args.bert_path, DICT_DATASET)

LABEL_SIZE = dataLoader.LABEL_SIZE

model = E2EMERN(args.bert_path, LABEL_SIZE)
optimizer = tf.keras.optimizers.Adam(1e-5)


def load_model(model, optimizer, data):
    (
        train_s_ind,
        train_s_seg,
        train_e_ind,
        train_e_seg,
        train_ner,
        train_cpt_ner,
        train_nen,
        _,
    ) = next(iter(data))
    train_one_step(
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
    model.load_weights(args.weights, by_name=True)


load_model(model, optimizer, dataLoader.Data("train"))

test_result = evaluate(model, dataLoader, "test")
test_ner_f1 = "%.4f" % test_result["ner"][-1]
test_nen_f1 = "%.4f" % test_result["nen"][-1]
print(f"Test: [NER] {test_ner_f1} [NEN] {test_nen_f1}")
if args.save_pred_result:
    save_prediction_result(model, dataLoader, f"results/test_eval.tsv", "test")

zs_test_result = evaluate(model, dataLoader, "zs_test")
zs_test_ner_f1 = "%.4f" % zs_test_result["ner"][-1]
if args.save_pred_result:
    save_prediction_result(model, dataLoader, f"results/zs_eval.tsv", "zs_test")
print(f"Test unseen samples: [NER] {zs_test_ner_f1}")