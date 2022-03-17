# Residual Neural Networks for the Prediction of Planetary Collision Outcomes

### Description

Official code for the paper "Residual Neural Networks for the Prediction of Planetary Collision Outcomes".

We provide a dataset that consists of 10164 Smooth Particle Hydrodynamics (SPH) simulations of pairwise planetary collisions. The data is available at https://phaidra.univie.ac.at/o:1206181.

![plot](./misc/github1.png)
  
We propose a weight-tied residual neural network for prediction of post-collision states. Our architecture incorporates an inductive bias to treat temporal dynamics consistently by evolving system states in an autoregressive manner. We outperform common baselines such as perfect inelastic merging and feed-forward neural networks.

![plot](./misc/github2.png)

### Usage

1) install used packages
2) set parameters in config.yaml. The "mode" variable defines the general usage. 
3) run script with `python3 main.py`. The CUDA device can be specified by the environment variable `CUDA_VISIBLE_DEVICES`.

**Data generation** was performed using CentOS Linux 7 with CUDA version 10.1. Initial conditions (i.e. the first frame) is produced by `SPH/spheres_ini/spheres_ini`, making use of SEAGen (https://github.com/jkeger/seagen) for initializing bodies. For the SPH simulation code, please visit https://github.com/christophmschaefer/miluphcuda. We provide our used parameter file in `SPH/miluphcuda/parameter.h`. Note that this parameter file may not be compatible with the newest miluphcuda version anymore. Tools for postprocessing can be found at 
`SPH/miluphcuda/utils/postprocessing/fast_identify_fragments_and_calc_aggregates/`.

**Machine Learning** was performed using Ubuntu 18.04 with pytorch version 1.9 and CUDA version 10.2. We provide two pre-trained models `misc/model_ema_ffn_pre-trained.pt` and `misc/model_ema_res_pre-trained.pt`, which can be integrated into N-body frameworks. Take care to apply correct pre -and postprocessing for model in -and outputs as in `dataloader.py`. Make sure that model inputs are valid, i.e. similar to samples in our dataset.

For questions please contact <winter@ml.jku.at> (Philip Winter).
