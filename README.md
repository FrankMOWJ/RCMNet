# ETSM: Automating Dissection Trajectory Suggestion and Confidence Map-Based Safety Margin Prediction for Robot-assisted Endoscopic Submucosal Dissection

## Prerequisite

Make sure that you have downloaded `PyTorch` and `MMsegmentation`.

## How to train

if you want to train the model, use the following command.

```shell
python train.py \
    --backbone_size base \
    --confidence_head segformer \
    --crop_size 532 \
    --batch_size 8 \
    --epoch 100 \
    --init_lr 0.001 \
    --save_dir ./log \
    --device cuda:0 
```

## Visualization

The visualization code is in `visualization.py`. Some important parameters are listed as follows.

```shell
python visualization.py \
    --data_root \PATH\TO\IMAGE\ROOT
    --checkpoint_path \PATH\TO\CHECKPOINT
    --dst_folder \PATH\TO\SAVE\RESULTS
```

`data_root` is the root of the dataset, by default is `./dataset`.

`dst_folder` is the destination folder to store the visualization results. If not provided, it will be set as `./visualization` in default.



