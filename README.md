# MAMNet: Multi-path Adaptive Modulation Network for Image Super-Resolution
![teaser_image](figures/teaser_image.png)


## Training

```shell
python main.py
  --device 0
  --model_name MAMNet
  --data_dir <path of the DIV2K dataset>
  --exp_dir <path of experiments>
  --exp_name <name of experiment> 
  --num_res 64 --num_feats 64 
  --is_MAM --is_CSI --is_ICD --is_CSD 
  --scale <scaling factor> 
  --is_init_res 
  --is_train 
```

## Test

``` shell
python main.py 
  --device 0 
  --model_name MAMNet 
  --data_dir <path of test datasets>
  --dataset_name <name of the test dataset>
  --exp_dir <path of experiments> 
  --exp_name <name of experiment>  
  --num_res 64 --num_feats 64 
  --is_MAM --is_CSI --is_ICD --is_CSD 
  --scale <scaling factor> 
  --is_test
```
