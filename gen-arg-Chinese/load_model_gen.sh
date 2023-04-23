# 根据实际checkpoints的路径修改CHECKPOINT_PATH
CHECKPOINT_PATH=checkpoints/epoch=2.ckpt 

python load_model_gen.py --model='gen' --load_ckpt=$CHECKPOINT_PATH