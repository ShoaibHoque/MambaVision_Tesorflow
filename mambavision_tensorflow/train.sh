
DATA_PATH="/ImageNet/train"
MODEL=mamba_vision_T
BS=2
EXP=Test
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25

# Set TensorFlow environment variables if needed
export TF_CPP_MIN_LOG_LEVEL=2  # Set TensorFlow logging level

# Run TensorFlow script
python train_tf.py --mesa ${MESA} --input_size 224 224 --crop_pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --batch_size $BS --tag $EXP --lr $LR --warmup_lr $WR_LR --weight_decay $WD --drop_path $DR
