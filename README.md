# Noise-Robust Bidirectional Learning with Dynamic Sample Reweighting

## Usage
### Environments

pip install -r requirements.txt

## Usage

1. Clone to the local.
```
> git clone https://github.com/chenchenzong/BLDR.git BLDR
```
2. Install required packages.
```
> cd BLDR
> pip install requirements.txt
```
3. Set parameters.
- Edit run.sh 
- dataset: cifar10/cifar100, 
- noise_type: aggre/rand1/worst for cifar10, noisy100 for cifar100.
4. Run the code. (The recognition results are named detection.npy, attention to file rewriting!)
```
> sh test.sh	# train the model
> python learning.py --dataset dataset --noise_type noise_type	# for image classification
> python detection.py --dataset dataset --noise_type noise_type	# for label noise detections