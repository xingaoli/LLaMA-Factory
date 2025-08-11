#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export FORCE_TORCHRUN=1
export TOKENIZERS_PARALLELISM=false
llamafactory-cli train train.yaml