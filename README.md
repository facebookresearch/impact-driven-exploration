# RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments

This is an implementation of the method proposed in <a href="https://openreview.net/pdf?id=rkg-TJBFPB">RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments</a>, which was published at ICLR 2020. The code includes all the baselines and ablations used in the paper. 

We propose a novel type of intrinsic reward which encourges the agent to take actions that result in significant changes to its representation of the environment state.

## Installation

```
# create a new conda environment
conda create -n ride-env python=3.6.8
conda activate ride-env 

# install dependencies
git clone git@github.com:fairinternal/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt
```

## Train RIDE on MiniGrid
```
cd impact-driven-exploration

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-ObstructedMaze-2Dlh-v0 

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-KeyCorridorS3R3-v0 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005
```

## Acknowledgements
Our vanilla RL algorithm is based on [Torchbeast](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
