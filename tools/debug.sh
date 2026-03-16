#!/usr/bin/env bash

TSA_DEBUG=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --eval bbox