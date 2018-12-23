#!/usr/bin/env bash
GPUID=1
export CUDA_VISIBLE_DEVICES=$GPUID

rootDir=''
date='1220'
aligns=("_homography") #"_homography" "_nowarp" "_OF"
modelName='model2_symskip_nngraph2_deeper'
epochTest=400 #don't know what this means

for align in ${aligns[@]}
do
    trainsetFolder="${rootDir}data/training_augumented_all_nostab${align}"
    validsetFolder="${rootDir}data/validating_augumented_all_nostab${align}"
#    testsetFolder="${rootDir}data/original_testing_real_all_nostab${align}"
    testsetFolder="/home/omnisky/Desktop/zc/DeepVideoDeblurring/data/training_real_all_nostab${align}"
    modelDir="${rootDir}models/${modelName}.py"
    paramSaveFolder="${rootDir}logs/${date}_${modelName}${align}"
#    outDir="${rootDir}outImg/${date}_${modelName}${align}"
    outDir="${rootDir}outImg/${date}_training_${modelName}${align}"

#    ############################
#    # train from the beginning #
#    ############################
#    python run_model.py \
#    --model ${modelDir} \
#    --data_trainset ${trainsetFolder} \
#    --ckp_dir ${paramSaveFolder} \
#    --num_gpus 1

    ######################
    # test using testset #
    ######################
    python run_model.py \
    --model ${modelDir} \
    --data_testset ${testsetFolder} \
    --ckp_dir ${paramSaveFolder} \
    --output_dir ${outDir}

#    ################################
#    # estimate loss using validset #
#    ################################
#    python run_model.py \
#    --model ${modelDir} \
#    --data_validset ${validsetFolder} \
#    --ckp_dir ${paramSaveFolder} \
done


