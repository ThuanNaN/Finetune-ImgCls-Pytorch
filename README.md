# Pytorch Image Classification from pre-trained

## Data folder
- All sub-folder contain image of one class will be in one folder "./data_dir" 
- dataset.py will automatically split to train/valid and test DataLoader

### Usage:
1. Intall requirement.txt
```
pip install -r requirements.txt
```
3. Train model

Example:
```
python train.py --load_checkpoint False --N_EPOCH 10 --model_name vgg16 --n_class 30
```
Note: Should be create new folder and named it "checkpoint" in main directory.
