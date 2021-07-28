# E2EMERN
[Title] An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization

[Authors] Baohang Zhou, Xiangrui Cai, Ying Zhang, Xiaojie Yuan

[ACL 2021 paper](https://aclanthology.org/2021.acl-long.485/)

## Preparation
1. Clone the repo to your local.
2. Download Python version: 3.6.5.
3. Download the pre-trained Bio-BERT models from this [link](https://github.com/dmis-lab/biobert). We use the *BioBERT-Large* in our experiments.
4. Open the shell or cmd in this repo folder. Run this command to install necessary packages.
```cmd
pip install -r requirements.txt
```

## Experiments
1. For Linux systems, we have shell scripts to run the training procedures. You can run the following command:
```cmd
./train.ncbi.sh or ./train.bc5cdr.sh
```

2. You can also input the following command to train the model. There are different choices for some parameters shown in square brackets. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
| epoch | int | Training times |
| LAMBDA | float | hyper-parameter in loss function |
| MU | float | hyper-parameter in loss function |
| bert_path | str | folder path of pre-trained BERT model |
| save_pred_result | bool | save the prediction result |

```cmd
python main.py \
    --seed 11 \
    --epoch 12 \
    --LAMBDA 0.125 \
    --MU 0.1 \
    --dataset [ncbi, cdr] \
    --bert_path ./biobert_large \
    --save_pred_result \
```

3. After training the model, the test result is saved in the "results" folder. And the weights of the model are saved in the "weights" folder.

4. We also provide the weights of the model to reimplement the results in our
paper. You can download the [weights file](https://pan.baidu.com/s/15DLSb2fvgbOiiv0V0ADFNg) (the extraction code **1234**) and put them into the "weights" folder. Then run the following command:
```cmd
./eval.ncbi.sh or ./eval.bc5cdr.sh
```

Bibtex:
```
@inproceedings{DBLP:conf/acl/ZhouC0Y20,
  author    = {Baohang Zhou and
               Xiangrui Cai and
               Ying Zhang and
               Xiaojie Yuan},
  title     = {An End-to-End Progressive Multi-Task Learning Framework for Medical
               Named Entity Recognition and Normalization},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
               Linguistics and the 11th International Joint Conference on Natural
               Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
               Event, August 1-6, 2021},
  pages     = {6214--6224},
  year      = {2020},
  crossref  = {DBLP:conf/acl/2021-1},
  url       = {https://aclanthology.org/2021.acl-long.485},
  timestamp = {Tue, 27 Jul 2021 12:03:20 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/ZhouC0Y20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
