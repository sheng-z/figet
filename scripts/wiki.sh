#!/bin/bash


set -o errexit

# Data
corpus_name=Wiki
corpus_dir=data/${corpus_name}
dataset_dir=${corpus_dir}/dataset

# Embeddings
embeddings_dir=data/embeddings
embeddings=${embeddings_dir}/glove.840B.300d.txt

# Checkpoints
ckpt=${corpus_dir}/ckpt

do_what=$1

mkdir -p ${corpus_dir}

if [ "${do_what}" == "get_data" ];
then
    printf "\nDownloading corpus...`date`\n"
    if [ -d "${corpus_dir}/dataset" ]; then
        echo "Seems that you already have the dataset!"
    else
        wget http://www.cs.jhu.edu/~s.zhang/data/figet/${corpus_name}.zip -O ${corpus_dir}/dataset.zip
        (cd ${corpus_dir} && unzip dataset.zip && rm dataset.zip)
    fi

    printf "\nDownloading word embeddings...`date`\n"
    if [ -d "${embeddings_dir}" ]; then
        echo "Seems that you already have the embeddings!"
    else
        mkdir -p ${embeddings_dir}
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ${embeddings_dir}/embeddings.zip
        (cd ${embeddings_dir} && unzip embeddings.zip && rm embeddings.zip)
    fi

elif [ "${do_what}" == "preprocess" ];
then
    mkdir -p ${ckpt}
    python -u ./preprocess.py \
        --train=${dataset_dir}/train.txt --dev=${dataset_dir}/dev.txt   \
        --test=${dataset_dir}/test.txt \
        --use_doc=0 --word2vec=${embeddings} \
        --save_data=${ckpt}/${corpus_name} --shuffle

elif [ "${do_what}" == "train" ];
then
    python -u ./train.py \
        --data=${ckpt}/${corpus_name}.data.pt \
        --word2vec=${ckpt}/${corpus_name}.word2vec \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_tuning=${ckpt}/${corpus_name}.tuning.pt \
        --niter=-1 --gpus=0 \
        --single_context=1 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=1 --bias=0 --context_length=10

elif [ "${do_what}" == "adaptive-thres" ];
then
    python -u -m figet.adaptive_thres \
        --data=${ckpt}/${corpus_name}.tuning.pt \
        --optimal_thresholds=${ckpt}/${corpus_name}.thres

elif [ "${do_what}" == "inference" ];
then
    python -u ./infer.py \
        --data=${dataset_dir}/test.txt \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_idx2threshold=${ckpt}/${corpus_name}.thres \
        --pred=${ckpt}/${corpus_name}.pred.txt --gpus=0 \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=1 --bias=0 --context_length=10
fi

