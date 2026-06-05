#!/usr/bin/env bash
set -e

DATASET=mnist
N_KERNELS=32
KERNEL_SIZE=3
HIDDEN_CHANNELS="32 64"
EPOCHS_TEACHER=60
EPOCHS_STUDENT=80
MLP_LAYERS=1

# MODE=1vsr
MODE=multiclass
TARGET_CLASS=0

if [ "$MODE" = "1vsr" ]; then
    SUFFIX="_class${TARGET_CLASS}"
else
    SUFFIX=""
fi

TEACHER_CKPT=ckpts/${DATASET}_teacher${SUFFIX}.pt
LOGITS_PATH=ckpts/${DATASET}_logits${SUFFIX}.pt
STUDENT_CKPT=ckpts/${DATASET}_student${SUFFIX}.pt

COMMON="--dataset $DATASET --n-kernels $N_KERNELS --kernel-size $KERNEL_SIZE --hidden-channels $HIDDEN_CHANNELS"


# printf "\nTraining teacher model..."
# if [ "$MODE" = "1vsr" ]; then
#     python scripts/train_teacher.py $COMMON \
#         --epochs $EPOCHS_TEACHER \
#         --mlp-layers $MLP_LAYERS \
#         --mode 1vsr \
#         --target-class $TARGET_CLASS \
#         --save-path $TEACHER_CKPT
# else
#     python scripts/train_teacher.py $COMMON \
#         --epochs $EPOCHS_TEACHER \
#         --mlp-layers $MLP_LAYERS \
#         --save-path $TEACHER_CKPT
# fi


# printf "\nCalibrating teacher model and generating logits for student training..."
# if [ "$MODE" = "1vsr" ]; then
#     python scripts/calibrate.py $COMMON \
#         --mlp-layers $MLP_LAYERS \
#         --mode 1vsr \
#         --target-class $TARGET_CLASS \
#         --teacher-ckpt $TEACHER_CKPT \
#         --logits-path $LOGITS_PATH
# else
#     python scripts/calibrate.py $COMMON \
#         --mlp-layers $MLP_LAYERS \
#         --teacher-ckpt $TEACHER_CKPT \
#         --logits-path $LOGITS_PATH
# fi


N_KERNELS=16
COMMON="--dataset $DATASET --n-kernels $N_KERNELS --kernel-size $KERNEL_SIZE --hidden-channels $HIDDEN_CHANNELS"

printf "\nTraining student model..."
if [ "$MODE" = "1vsr" ]; then
    python scripts/train_student.py $COMMON \
        --epochs $EPOCHS_STUDENT \
        --mode 1vsr \
        --target-class $TARGET_CLASS \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT
else
    python scripts/train_student.py $COMMON \
        --epochs $EPOCHS_STUDENT \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT
fi
