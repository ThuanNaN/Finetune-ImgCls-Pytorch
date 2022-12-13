# Finetune Pytorch model for Image Classification


## 1. Data Structure

```
data_dir/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...
```

## 2. Install packages
```
pip3 install -r requirements.txt
```

## 3. Config 
### 3.1 Config data training
Setting it in config/data_config.yaml

Example:
```
train: ./data/fold_0/train
val: ./data/fold_0/val
test: ./data/fold_0/test
n_classes: 2
```

### 3.2 Config model training
Setting it in config/train_config.yaml
Example:
```
n_epochs: 5
batch_size: 16
learning_rate: 0.0002
weight_decay: 0.0002
PATH_SAVE: ./ckpt
model_name: resnet50
weight_name: IMAGENET1K_V1
load_weight: True
freeze_backbone: False
default_data_transform: False
```

## 4. Run
To train model, just run this command:
```
python train.py
```
