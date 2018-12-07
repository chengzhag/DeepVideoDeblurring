#!/usr/bin/env bash
GPUID=0,1
export CUDA_VISIBLE_DEVICES=$GPUID

rootDir=''
date='1207'
align="_nowarp"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

trainsetFolder="${rootDir}data/training_augumented_all_nostab${align}"
validsetFolder="${rootDir}data/validating_augumented_all_nostab${align}"
testsetFolder="${rootDir}data/testing_real_all_nostab_nowarp"
modelDir="${rootDir}models/${modelName}.py"
paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"
outDir="${rootDir}outImg/${date}_${modelName}${align}"


############################
# train from the beginning #
############################
python run_model.py \
--model ${modelDir} \
--data_trainset ${trainsetFolder} \
--ckp_dir ${paramSaveFolder} \
--num_gpus 2

#######################
## test using testset #
#######################
#python run_model.py \
#--model ${modelDir} \
#--data_testset ${testsetFolder} \
#--ckp_dir ${paramSaveFolder} \
#--output_dir ${outDir}

#################################
## estimate loss using validset #
#################################
#python run_model.py \
#--model ${modelDir} \
#--data_validset ${validsetFolder} \
#--ckp_dir ${paramSaveFolder} \



