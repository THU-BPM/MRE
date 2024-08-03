DATASET_NAME="MRE"
BERT_NAME="bert-base-uncased"

CUDA_VISIBLE_DEVICES=5 nohup    python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=5e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --use_prompt \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --caption \
        --use_attention\
        --notes='caption'\
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path='ckpt/re/'&