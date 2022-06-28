# Noise-Robust Bidirectional Learning with Dynamic Sample Reweighting

A Submission for [LMNL 2022](http://ucsc-real.soe.ucsc.edu:1995/Competition.html)(1st Learning and Mining with Noisy Labels Challenge IJCAI-ECAI 2022).

## Usage
### Environments
```
> pip install -r requirements.txt
```
## Usage

1. Clone to the local.
```
> git clone https://github.com/chenchenzong/BLDR.git BLDR
```
2. Install required packages.
```
> cd BLDR
> mkdir ckpts
> mkdir dataset
> pip install requirements.txt
```
3. Set parameters.
- Edit run.sh 
- For cifar 10 aggre/rand1/worst
```
> nohup python3 train.py --dataset cifar10 --noise_type worst  > c10_worst.log
> nohup python3 train.py --dataset cifar10 --noise_type rand1  > c10_rand1.log
> nohup python3 train.py --dataset cifar10 --noise_type aggre  > c10_aggre.log
```
- For cifar 100 noisy
```
> nohup python3 train.py --dataset cifar100 --noise_type noisy100 --updateW_epochs 40  > c100.log
```
4. Run the code. (The recognition results are named detection.npy, attention to file rewriting!)
```
> sh test.sh	# train the model
> python learning.py --dataset dataset --noise_type noise_type	# for image classification
> python detection.py --dataset dataset --noise_type noise_type	# for label noise detections
```
