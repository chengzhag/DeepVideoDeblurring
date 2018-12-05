#!/usr/bin/env bash

############################
# train from the beginning #
############################
rootDir=''
date='1205'
align="_nowarp"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

trainsetFolder="${rootDir}data/training_augumented_croped_all_nostab${align}"
validsetFolder="${rootDir}data/validating_augumented_croped_all_nostab${align}"
modelDir="${rootDir}models/${modelName}.py"
paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"

python run_model.py \
--model ${modelDir} \
--data_trainset ${trainsetFolder} \
--ckp_dir ${paramSaveFolder} \
