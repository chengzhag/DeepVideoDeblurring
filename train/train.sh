#!/bin/bash
# Params
rootDir='../'
align="_nowarp"
alignFolder="${rootDir}data/training_augumented_all_nostab${align}"
modelDir=${rootDir}'models/model2_symskip_nngraph2_deeper.lua'

GPUID=0
export CUDA_VISIBLE_DEVICES=$GPUID

th train.lua \
--data_root ${alignFolder} \
--model ${modelDir}
