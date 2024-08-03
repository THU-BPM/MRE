#!/usr/bin/env bash

DATASET_NAME="MRE"
BERT_NAME="bert-base-uncased"

CUDA_VISIBLE_DEVICES=4  nohup  python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=6e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --notes='full'\
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --caption\
        --save_path='ckpt/re/' &
CUDA_VISIBLE_DEVICES=2 nohup  python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=3e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --notes='full'\
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path='ckpt/re/' &
CUDA_VISIBLE_DEVICES=3 nohup  python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=3e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --caption\
        --notes='full'\
        --save_path='ckpt/re/'&
 CUDA_VISIBLE_DEVICES=1 nohup  python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=4e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --caption\
        --notes='full'\
        --save_path='ckpt/re/'&
CUDA_VISIBLE_DEVICES=0 nohup  python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=5e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --notes='full'\
        --caption\
        --save_path='ckpt/re/'&