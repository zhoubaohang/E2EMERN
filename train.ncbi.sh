#!/bin/bash
echo Searching hyper-parameters...
eval $(\
    python search_hp.py \
    --seed 11 \
    --epoch 20 \
    --LAMBDA 0.125 \
    --MU 0.1 \
    --dataset ncbi \
    --bert_path ./biobert_large \
    | awk '{printf("epoch=%s; lambda=%s; mu=%s;",$1,$2,$3)}')
echo Training the models...
python main.py \
    --seed 11 \
    --epoch ${epoch} \
    --LAMBDA ${lambda} \
    --MU ${mu} \
    --dataset ncbi \
    --save_weights \
    --bert_path ./biobert_large \