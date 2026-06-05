#!/usr/bin/env bash
set -e

DATASET=mnist
N_KERNELS=16
KERNEL_SIZE=3
HIDDEN_CHANNELS="32 64"
EPOCHS_STUDENT=80

# MODE=1vsr
MODE=multiclass
TARGET_CLASS=0

if [ "$MODE" = "1vsr" ]; then
    SUFFIX="_class${TARGET_CLASS}"
else
    SUFFIX=""
fi

LOGITS_PATH=ckpts/${DATASET}_logits${SUFFIX}.pt
STUDENT_CKPT=ckpts/${DATASET}_linear_student${SUFFIX}.pt

COMMON="--dataset $DATASET --n-kernels $N_KERNELS --kernel-size $KERNEL_SIZE --hidden-channels $HIDDEN_CHANNELS"

printf "\nTraining linear student model..."
if [ "$MODE" = "1vsr" ]; then
    python scripts/train_student.py $COMMON \
        --epochs $EPOCHS_STUDENT \
        --mode 1vsr \
        --target-class $TARGET_CLASS \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT \
        --model linear
else
    python scripts/train_student.py $COMMON \
        --epochs $EPOCHS_STUDENT \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT \
        --model linear
fi
