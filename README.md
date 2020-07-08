# Dependencies

The following dependencies are required:
```
gpytorch==1.0.1
numpy==1.17.4
pypolyagamma==1.2.2
sacred==0.8.0
tensorboardX==1.9
torch==1.4.0
tqdm==4.38.0
```
Other versions of the above packages may work but have not been tested.

# Downloading Data

Scripts for downloading each dataset can be found in `filelists` under `download_X.sh`, where `X` is the dataset.

# Training

Training is handled by `train.py`. Options can be found by running `python train.py print_config`. For example, to train on CUB:

`python train.py with method=ove_polya_gamma_gp dataset=CUB train_aug=True kernel.name=L2LinearKernel num_draws=20 num_steps=1 n_shot=5`

This will train a cosine kernel, marginal likelihood model on CUB. For predictive likelihood, use `method=predictive_ove_polya_gamma_gp`.

# Testing

Testing is handled by `test.py`. Options can be found by running `python test.py print_config`. When testing, you will need to specify the `job_id` of the training run you would like to evaluate (by default saved in `runs`). For example, to evaluate job 1 on 5-way, 5-shot:

`python test.py with job_id=1 n_shot=5 test_n_way=5 kernel.name=L2LinearKernel num_draws=20 num_steps=50`

Note that manual specification of the kernel, num_draws (parallel Gibbs chains) and `num_steps` (length of Gibbs chains) is required.

# Our Model

The implementation of our model can be found in `methods/ove_polya_gamma.py`. The function `set_forward_loss` returns the training loss for an episode and `set_forward` returns predicted logits.

# Acknowledgements

This code is forked from https://github.com/BayesWatch/deep-kernel-transfer, which itself is a fork of https://github.com/wyharveychen/CloserLookFewShot.
