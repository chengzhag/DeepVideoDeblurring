#!/bin/bash

GPUID=0
export CUDA_VISIBLE_DEVICES=$GPUID

# ########################################
# # overfit 1 patch from 80000 its model #
# ########################################
# rootDir='../'
# date='1120'
# dateLoad='1112'
# itLoadFix='_iteration_80000'
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
# --it_max 2000 \
# --save_every 1000 \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --reset_lr 0.005 \
# --reset_state 1 \
# --decay_from 2000 \
# --decay_every 500 \
# --overfit_batches 1 \
# --overfit_out "${rootDir}outImg/${date}_${modelName}${align}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \
# --overfit_patches 1 \

# ########################################
# # overfit 1 batch from 80000 its model #
# ########################################
# rootDir='../'
# date='1120'
# dateLoad='1112'
# itLoadFix='_iteration_80000'
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
# --it_max 2000 \
# --save_every 1000 \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --reset_lr 0.005 \
# --reset_state 1 \
# --decay_from 2000 \
# --decay_every 500 \
# --overfit_batches 1 \
# --overfit_out "${rootDir}outImg/${date}_${modelName}${align}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \


# ###############################################
# # continue overfitting 1 batch from old model #
# ###############################################
# rootDir='../'
# date='1120'
# dateLoad='1120'
# itLoadFix='_iteration_5000'
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
# --it_max 10000 \
# --save_every 1000 \
# --model_param_load "${paramLoadFolder}/param_epoch_${epochTest}${itLoadFix}.t7" \
# --bn_meanstd_load "${paramLoadFolder}/bn_meanvar_epoch_${epochTest}${itLoadFix}.t7" \
# --optimstate_load "${paramLoadFolder}/optimstate_epoch_${epochTest}${itLoadFix}.t7" \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --decay_from 10000 \
# --decay_every 500 \
# --overfit_batches 1 \
# --overfit_out "${rootDir}outImg/${date}_${modelName}${align}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \
# --reset_lr 0.005 \

# ####################################
# # overfit n batches from new model #
# ####################################
# rootDir='../'
# date='1121'
# dateLoad='1112'
# itLoadFix='_iteration_80000'
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
# --it_max 800 \
# --save_every 1000 \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --reset_lr 0.005 \
# --reset_state 1 \
# --decay_from 1000 \
# --decay_every 200 \
# --decay_rate 0.1 \
# --overfit_batches 20 \
# --overfit_out "${rootDir}outImg/${date}_${modelName}${align}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \

# #####################################################
# # overfit n batches from new model inited by nninit #
# #####################################################
# rootDir='../'
# date='1121'
# dateLoad='1112'
# itLoadFix='_iteration_80000'
# align="_nowarp"
# modelName='model2_symskip_nngraph2_deeper_nninit'
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
# --it_max 800 \
# --save_every 1000 \
# --model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
# --bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
# --optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
# --reset_lr 0.005 \
# --reset_state 1 \
# --decay_from 1000 \
# --decay_every 200 \
# --decay_rate 0.1 \
# --overfit_batches 20 \
# --overfit_out "${rootDir}outImg/${date}_${modelName}${align}" \
# --log "${paramSaveFolder}/train.log" \
# --log_every 10 \

#######################################################
# overfit n batches from new model using new datasets #
#######################################################
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
paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}_overfit"

th train.lua \
--seed 251 \
--model ${modelDir} \
--data_trainset ${trainsetFolder} \
--data_validset ${validsetFolder} \
--batch_size 64 \
--it_max 2000 \
--save_every 500 \
--model_param "${paramSaveFolder}/param_epoch_${epochTest}" \
--bn_meanstd "${paramSaveFolder}/bn_meanvar_epoch_${epochTest}" \
--optimstate "${paramSaveFolder}/optimstate_epoch_${epochTest}" \
--reset_lr 0.005 \
--reset_state 1 \
--decay_from 500 \
--decay_every 300 \
--decay_rate 0.1 \
--overfit_batches 20 \
--overfit_out "${rootDir}outImg/${date}_${modelName}${align}_overfit" \
--log "${paramSaveFolder}/train.log" \
--log_every 5 \

