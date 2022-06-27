CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --dataset cifar10 --noise_type worst  > c10_worst.log &  

CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --dataset cifar10 --noise_type rand1  > c10_rand1.log &  

CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --dataset cifar10 --noise_type aggre  > c10_aggre.log & 

CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --dataset cifar100 --noise_type noisy100  > c100.log & 