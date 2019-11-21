#!/usr/bin/env bash


while :
do
expressions=("0.1" "0.3")
alpha=${expressions[$RANDOM % ${#expressions[@]} ]}
echo $alpha


expressions=("4e-4" "4e-5")
lrg=${expressions[$RANDOM % ${#expressions[@]} ]}
echo $lrg

expressions=("True" "False")
rand=${expressions[$RANDOM % ${#expressions[@]} ]}
echo $rand

topk=`shuf -i 2-15 -n 1`
echo $topk


CUDA_VISIBLE_DEVICES=1 python main.py --gpus 1 --data_root /local/pajot/data/proba_v/train --batch_size 16 --alpha $alpha --lrg $lrg --nc $topk --topk $topk --rand $rand --stat True
done