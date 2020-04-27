### Updates
- Apr21: Fixed a few bugs and update readme.
- Apr24: Update config 
- Apr25: add voc mAP metric

### 1. efficientdet-tf2
[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
Arxiv link: https://arxiv.org/abs/1911.09070
[2] https://github.com/google/automl

This is the tf2.0 version of efficientdet.

### 2. Pretrained EfficientDet Checkpoints
The checkpoints and results is [here](https://github.com/google/automl/tree/master/efficientdet).

### 3. Saved model
```
python3 -m inferences.efficientdet --input_size=512x512
```
Note! We should add the checkpoints to pretrained_weights. The default model is efficientdet-d0, if you want to use others, you should modify the configs/efficiendet_configs.py.

The new efficientdet-d0 implementation run around 26ms, faster than official TF version, because we use combined_non_maximum instead the official version NMS (the input size is 512x512, the official efficientdet-d0 is 1280x1920). *Note, run this test on P4000 GPU, ubuntu 18.04.*

### 4. Tensorrt
```
python3 -m inferences.efficientdet --mode=FP16 --saved_model_dir=./saved_model/efficientdet-d0/1  --output_dir=./trt_model/efficientdet-d0/1
```
Note, only support FP16 and FP32.

### 4. Run demo
```
python3 demo.py --saved_model ./saved_model/efficientdet-d0/1 --video_path xxx.mp4
```

