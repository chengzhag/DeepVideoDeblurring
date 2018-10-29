#!/bin/bash
# Params
rootDir='../'
date='1029'
align="_nowarp"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

alignFolder="${rootDir}data/training_augumented_all_nostab${align}"
modelDir="${rootDir}models/${modelName}.lua"

GPUID=0
export CUDA_VISIBLE_DEVICES=$GPUID

th train.lua \
--data_root ${alignFolder} \
--model ${modelDir} \
--batch_size 64 \
--model_param "${rootDir}logs/${date}_${modelName}${align}/param_epoch_${epochTest}" \
--bn_meanstd "${rootDir}logs/${date}_${modelName}${align}/bn_meanvar_epoch_${epochTest}" \
--save_every 3000 \
--load_save 0 \
--log_save "${rootDir}logs/${date}_${modelName}${align}/log.log" \
--seed 251

