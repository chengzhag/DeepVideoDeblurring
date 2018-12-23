# ####################
# predict on testset #
# ####################
#!/bin/bash
# Params
ALIGNMENT=("_nowarp") # "_OF" "_homography")

MODEL="model2_symskip_nngraph2_deeper"
METHOD=""
DATE="1130"
itFix='_iteration_80000'
FNFix=''
for align in ${ALIGNMENT[@]}
# for((i=0;i<${#ALIGNMENT[@]};i++))
do
    FN=${DATE}"_"$MODEL${align}$METHOD
    echo 'FN='${FN}
    FN_ALIGNS=(${FN_ALIGNS[@]} ${FN})
    REALSET_ALIGNS=(${REALSET_ALIGNS[@]} "data/testing_real_all_nostab"${align})
    FNREAL_ALIGNS=(${FNREAL_ALIGNS[@]} ${FN}"_real")
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
        echo "Processing "${subset}${ALIGNMENT[i]}"... "
        echo 'FN='${FN}
        echo 'REALSET='${REALSET}
        echo 'FNREAL='${FNREAL}
        th inference.lua -g $GPUID --model $MODEL --data_root $REALSET/$subset \
        --model_param logs/$FN$FNFix/param_epoch_$EPOCHTEST$itFix.t7 \
        --bn_meanstd logs/$FN$FNFix/bn_meanvar_epoch_$EPOCHTEST$itFix.t7 \
        --saveDir outImg/$FNREAL$FNFix$itFix/$subset \
        --start_id 1 --n 100 --patchbypatch 1 --patch_size 256
    done
done

# #####################
# predict on validset #
# #####################
# #!/bin/bash
# # Params
# ALIGNMENT=("_nowarp" "_OF" "_homography") #("_nowarp" "_OF" "_homography")
# validset=('IMG_0030' 'IMG_0049' 'IMG_0021' '720p_240fps_2' 'IMG_0032' \
# 'IMG_0033' 'IMG_0031' 'IMG_0003' 'IMG_0039' 'IMG_0037')

# MODEL="model2_symskip_nngraph2_deeper"
# METHOD=""
# DATE="original"
# itFix=''
# FNFix=''
# for align in ${ALIGNMENT[@]}
# # for((i=0;i<${#ALIGNMENT[@]};i++))
# do
#     FN=${DATE}"_"$MODEL${align}$METHOD
#     echo 'FN='${FN}
#     FN_ALIGNS=(${FN_ALIGNS[@]} ${FN})
#     REALSET_ALIGNS=(${REALSET_ALIGNS[@]} "data/training_real_all_nostab"${align})
#     FNREAL_ALIGNS=(${FNREAL_ALIGNS[@]} ${FN}"_real")
# done
# GPUID=0
# EPOCHTEST=400

# export CUDA_VISIBLE_DEVICES=$GPUID

# # Scan input video folders
# for subset in ${validset[@]}
# do
#     echo "Processing "${subset}"... "
#     # prediction on qualitative (real) set
#     for((i=0;i<${#ALIGNMENT[@]};i++))
#     do
#         FN=${FN_ALIGNS[i]}; REALSET=${REALSET_ALIGNS[i]}; FNREAL=${FNREAL_ALIGNS[i]}
#         echo "Processing "${subset}${ALIGNMENT[i]}"... "
#         echo 'FN='${FN}
#         echo 'REALSET='${REALSET}
#         echo 'FNREAL='${FNREAL}
#         th inference.lua -g $GPUID --model $MODEL --data_root $REALSET/$subset \
#         --model_param logs/$FN$FNFix/param_epoch_$EPOCHTEST$itFix.t7 \
#         --bn_meanstd logs/$FN$FNFix/bn_meanvar_epoch_$EPOCHTEST$itFix.t7 \
#         --saveDir outImg/${DATE}"_validating_"$MODEL${ALIGNMENT[i]}$METHOD$FNFix$itFix/$subset \
#         --start_id 1 --n 100 --patchbypatch 1 --patch_size 256
# #        --saveDir outImg/$FNREAL$FNFix$itFix/$subset \
#     done
# done


