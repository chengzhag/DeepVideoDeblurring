#!/bin/bash
# Params
rootDir='../'
date='1107'
dateLoad='1108'
itLoadFix=''
align="_nowarp"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

dataFolder="${rootDir}data/training_augumented_all_nostab${align}"
modelDir="${rootDir}models/${modelName}.lua"
paramLoadFolder="${rootDir}logs/${dateLoad}_${modelName}${align}"
paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"

GPUID=1
export CUDA_VISIBLE_DEVICES=$GPUID

th train.lua \
--seed 251 \
--model ${modelDir} \
--data_root ${dataFolder} \
--trainset_size 61 \
--batch_size 64 \
--it_max 80000 \
--model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
--bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
--optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
--save_every 1000 \
--log "${paramSaveFolder}/train.log" \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
