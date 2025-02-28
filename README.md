# ETSM: Automating Dissection Trajectory Suggestion and Confidence Map-Based Safety Margin Prediction for Robot-assisted Endoscopic Submucosal Dissection

## Prerequisite

Make sure that you have downloaded `PyTorch` and `MMsegmentation`.

## Prepare dataset
Send email to mengyaxu@cuhk.edu.hk to request to download our **ESTM** dataset from [here](https://drive.google.com/file/d/1RQf1q1c0dXzSFP_XO8Al3JTWsxsD1F95/view?usp=drive_link) and unzip it under the root directory of the project by

```shell
unzip ESTM.zip -d dataset
```

After unzipping, the folder structure should be as follows.

```tree
.
├── arch
├── dataset
├── log
├── utils
└── ...
```

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



