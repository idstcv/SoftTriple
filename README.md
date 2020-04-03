# SoftTriple Loss
PyTorch Implementation for ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

## Usage: Train on Cars196
Here is an example of using this package.

1. Obtain dataset
```
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar -xf car_ims.tgz
```

2. Generate train/test sets
```
python genCars.py
```

3. Learn 64-dimensional embeddings
```
python train.py --gpu 0 --dim 64 -C 98 --freeze_BN [folder with train and test folders]
```

## Requirements
* Python 3.7
* PyTorch 1.1
* scikit-learn 0.20.1

    
## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2019striple,
  author    = {Qi Qian and
               Lei Shang and
               Baigui Sun and
               Juhua Hu and
               Hao Li and
               Rong Jin},
  title     = {SoftTriple Loss: Deep Metric Learning Without Triplet Sampling},
  booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2019},
  year      = {2019}
}
```
