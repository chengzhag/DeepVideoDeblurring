#!/bin/bash
# Params
ALIGNMENT=("_OF" "_nowarp" "_homography")

MODEL="model2_symskip_nngraph2_deeper"
METHOD=""
DATE="1018"
for align in ${ALIGNMENT[@]}
do
    FN=${DATE}"_"$MODEL${align}$METHOD
    FN_ALIGNS=(${FN[@]} ${FN})
    REALSET_ALIGNS=(${REALSET[@]} "data/testing_real_all_nostab"${align})
    FNREAL_ALIGNS=(${FNREAL[@]} ${FN}"_real")
done
GPUID=0
EPOCHTEST=400

export CUDA_VISIBLE_DEVICES=$GPUID

# Scan input video folders
for subset in $(ls ${REALSET_ALIGNS})
do
    echo "Processing "${subset}"... "
    # prediction on qualitative (real) set
    for((i=0;i<${#ALIGNMENT[@]};i++))
    do
        FN=${FN_ALIGNS[i]}; REALSET=${REALSET_ALIGNS[i]}; FNREAL=${FNREAL_ALIGNS[i]}
        # echo ${FN}
        # echo ${REALSET}
        # echo ${FNREAL}
        th inference.lua -g $GPUID --model $MODEL --data_root $REALSET/$subset \
        --model_param logs/$FN/param_epoch_$EPOCHTEST.t7 \
        --bn_meanstd logs/$FN/bn_meanvar_epoch_$EPOCHTEST.t7 \
        --saveDir outImg/$FNREAL/$subset \
        --start_id 1 --n 100 --patchbypatch 1 --patch_size 256
    done
done


