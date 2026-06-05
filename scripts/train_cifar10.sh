#!/usr/bin/env bash
set -e

DATASET=cifar10
EPOCHS_TEACHER=100
EPOCHS_STUDENT=100
# Teacher (PreActResNet18) args
BASE_CHANNELS=64
# Student args
N_KERNELS=64
KERNEL_SIZE=3
HIDDEN_CHANNELS="64 128 256 128"
NIMO_HIDDEN_DIM=64
LAMBDA_REG=1.0
MU_REG=1.0

# MODE=1vsr
MODE=multiclass
TARGET_CLASS=3

if [ "$MODE" = "1vsr" ]; then
    SUFFIX="_class${TARGET_CLASS}"
else
    SUFFIX=""
fi

TEACHER_CKPT=ckpts/${DATASET}_teacher${SUFFIX}.pt
LOGITS_PATH=ckpts/${DATASET}_logits${SUFFIX}.pt
STUDENT_CKPT=ckpts/${DATASET}_student${SUFFIX}.pt

TEACHER_COMMON="--dataset $DATASET --arch preact_resnet --base-channels $BASE_CHANNELS"
STUDENT_COMMON="--dataset $DATASET --n-kernels $N_KERNELS --kernel-size $KERNEL_SIZE --hidden-channels $HIDDEN_CHANNELS --nimo-hidden-dim $NIMO_HIDDEN_DIM --lambda-reg $LAMBDA_REG --mu-reg $MU_REG"

# printf "Training teacher model...\n"
# if [ "$MODE" = "1vsr" ]; then
#     python scripts/train_teacher.py $TEACHER_COMMON \
#         --epochs $EPOCHS_TEACHER \
#         --mode 1vsr \
#         --target-class $TARGET_CLASS \
#         --save-path $TEACHER_CKPT
# else
#     python scripts/train_teacher.py $TEACHER_COMMON \
#         --epochs $EPOCHS_TEACHER \
#         --save-path $TEACHER_CKPT
# fi


# printf "\nCalibrating teacher model and generating logits for student training...\n"
# if [ "$MODE" = "1vsr" ]; then
#     python scripts/calibrate.py $TEACHER_COMMON \
#         --mode 1vsr \
#         --target-class $TARGET_CLASS \
#         --teacher-ckpt $TEACHER_CKPT \
#         --logits-path $LOGITS_PATH
# else
#     python scripts/calibrate.py $TEACHER_COMMON \
#         --teacher-ckpt $TEACHER_CKPT \
#         --logits-path $LOGITS_PATH
# fi


printf "\nTraining student model...\n"
if [ "$MODE" = "1vsr" ]; then
    python scripts/train_student.py $STUDENT_COMMON \
        --epochs $EPOCHS_STUDENT \
        --mode 1vsr \
        --target-class $TARGET_CLASS \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT
else
    python scripts/train_student.py $STUDENT_COMMON \
        --epochs $EPOCHS_STUDENT \
        --logits-path $LOGITS_PATH \
        --save-path $STUDENT_CKPT
fi
