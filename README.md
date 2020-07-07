# NPAttack_ECCV2020

This is our Pytorch implementation of NPAttack.

**Improving Query Efficiency of Black-box Adversarial Attack (ECCV2020)**

## Pre-trained model

You can download the pre-trained NP model for MNIST, CIFAR and ImageNet from https://drive.google.com/file/d/1TysxLn1SdVlPuwATPwSmq0T141oGqlRP/view?usp=sharing and put them into the folder of ./np_pretrained

The pre-trained target model in our experiments are available in https://drive.google.com/file/d/1uN22WfasesNfotAMVHCJ-9KjVh0bWdeP/view?usp=sharing , you can downloads them and put them into the folder of ./target_model or train you own models 

(Noted that if you train your own model, please be sure the input images are normalized  to [-0.5, 0.5] so as to match the normalization method of NP model ).

## NP model pre-training  

1. NP model for MNIST

   ```python
   CUDA_VISIBLE_DEVICES=0 python ./ANP/ANP_MNIST.py 
   ```

2. NP model for CIFAR10

   ```
   CUDA_VISIBLE_DEVICES=0 python ./ANP/ANP_CIFAR_train.py
   ```

3. NP model for ImageNet

   Specify the directory path of ImageNet dataset in ./ANP/ANP_IMAGENET.py

   ```
   CUDA_VISIBLE_DEVICES=0 python ./ANP/ANP_IMAGENET.py --data-path xxx
   ```

## Untargeted Attack

1. Untargeted Black box attack on MNIST

   Change the path of data in mnist_args in run_untargeted.py

   ```
   'data_path': #FIXME
   ```

   Specify the arguments for MNIST in run_untargeted.py

   ```
   args = mnist_args
   ```

   Run the file run_untargeted.py

   ```
   python run_untargeted.py
   ```

2. Untargeted Black box attack on CIFAR10

   Specify the arguments for CIFAR-10 in run_untargeted.py

   ```
   args = cifar_args
   ```

   Run the file run_untargeted.py

   ```
   python run_untargeted.py
   ```

## Targeted Attack

1. Specify the arguments for MNIST and CIFAR10 respectively in run_targeted.py as above.

2. Run the file run_targeted.py

   ```
   python run_targeted.py
   ```

## NPAttack on ImageNet

At least two GPUs are needed to run NPAttack on ImageNet

1. Untargeted Attack

   ```
   CUDA_VISIBLE_DEVICES=0,1 python NPAttack_IMAGENET.py --data-path xxx --type R
   ```

2. Targeted Attack

   ```
   CUDA_VISIBLE_DEVICES=0,1 python NPAttack_IMAGENET.py --data-path xxx --type R --targeted
   ```



