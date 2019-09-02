# MAMNet-tensorflow
Tensorflow implementation of MAMNet

## Training

```
python main.py --device 0 --model_name MAMNet --data_dir D:/Dataset/DIV2K --exp_dir D:/tensorflow/experiments --exp_name MAMNet_R16C64_X2 --num_res 16 --num_feats 64 --is_MAM --is_CSI --is_ICD --is_CSD --scale 2 --is_init_res --is_train 
```

## Test

```
python main.py --device 0 --model_name MAMNet --data_dir D:/Dataset/SR_test --exp_dir D:/tensorflow/experiments --exp_name MAMNet_R16C64_X2 --num_res 16 --num_feats 64 --is_MAM --is_CSI --is_ICD --is_CSD --scale 2 --is_test --dataset_name Set5 
```
