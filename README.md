# RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments

This is an implementation of the method proposed in <a href="https://openreview.net/pdf?id=rkg-TJBFPB">RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments</a>, which was published at ICLR 2020. The code includes all the baselines and ablations used in the paper. 

We propose a novel type of intrinsic reward which encourges the agent to take actions that result in significant changes to its representation of the environment state.

## Installation

```
# create a new conda environment
conda create -n ride-env python=3.6.8
conda activate ride-env 

# install PyTorch 
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

# install other requirements
git clone git@github.com:fairinternal/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt

# install the MiniGrid environments 
cd impact-driven-exploration
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install -e .
```

## Train RIDE on MiniGrid
```
cd impact-driven-exploration

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N12-S10-v0 

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N10-S4-v0 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005
```

## Acknowledgements
Our vanilla RL algorithm is based on [Torchbesat](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
