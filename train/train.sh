#!/bin/bash

GPUID=3
export CUDA_VISIBLE_DEVICES=$GPUID

# ##########################
# # train from n its model #
# ##########################
# rootDir='../'
# date='1124'
# dateLoad='1108'
# itLoadFix=''
# align="_nowarp"
# modelName='model2_symskip_nngraph2_deeper'
# epochTest=400 #don't know what this means

# dataFolder="${rootDir}data/training_augumented_all_nostab${align}"
# modelDir="${rootDir}models/${modelName}.lua"
# paramLoadFolder="${rootDir}logs/${dateLoad}_${modelName}${align}"
# paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"

# th train.lua \
# --seed 251 \
# --model ${modelDir} \
# --data_root ${dataFolder} \
# --trainset_size 61 \
# --batch_size 64 \
# --it_max 160000 \
# --save_every 1000 \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \


# ################################
# # finetune from original model #
# ################################
# rootDir='../'
# date='1124'
# dateLoad='1124'
# itLoadFix='_iteration_100000'
# align="_nowarp"
# modelName='model2_symskip_nngraph2_deeper'
# epochTest=400 #don't know what this means

# dataFolder="${rootDir}data/training_augumented_all_nostab${align}"
# modelDir="${rootDir}models/${modelName}.lua"
# paramLoadFolder="${rootDir}logs/${dateLoad}_${modelName}${align}_1e-5"
# paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"

# th train.lua \
# --seed 251 \
# --model ${modelDir} \
# --data_root ${dataFolder} \
# --trainset_size 61 \
# --batch_size 64 \
# --it_max 160000 \
# --save_every 1000 \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --reset_lr 0.00001 \
# --decay_from 100000 \
# --decay_every 200 \
# --decay_rate 0.1 \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \

###################################################
# finetune from original model using new datasets #
###################################################
rootDir='../'
date='1128'
dateLoad='1108'
itLoadFix=''
align="_nowarp"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

trainsetFolder="${rootDir}data/training_augumented_croped_all_nostab${align}"
validsetFolder="${rootDir}data/validating_augumented_croped_all_nostab${align}"
modelDir="${rootDir}models/${modelName}.lua"
paramLoadFolder="${rootDir}logs/${dateLoad}_${modelName}${align}"
paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"

th train.lua \
--seed 251 \
--model ${modelDir} \
--data_trainset ${trainsetFolder} \
--data_validset ${validsetFolder} \
--batch_size 64 \
--it_max 40000 \
--save_every 1000 \
--model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
--bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
--optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
--model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
--bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
--optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
--reset_lr 0.00001 \
--reset_state 1 \
--decay_from 15000 \
--decay_every 100000 \
--decay_rate 0.1 \
--log "${paramSaveFolder}/train.log" \
--log_every 10 \
