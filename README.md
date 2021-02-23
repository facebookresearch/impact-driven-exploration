# RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments

This is an implementation of the method proposed in 

<a href="https://openreview.net/pdf?id=rkg-TJBFPB">RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments</a> 

by Roberta Raileanu and Tim Rocktäschel, published at ICLR 2020. 

We propose a novel type of intrinsic reward which encourges the agent to take actions that result in significant changes to its representation of the environment state.

The code includes all the baselines and ablations used in the paper. 

The code was also used to run the baselines in [Learning with AMIGO:
Adversarially Motivated Intrinsic Goals](https://arxiv.org/pdf/2006.12122.pdf). 
See [the associated repo](https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals) for instructions on how to reproduce the results from that paper.

## Citation
If you use this code in your own work, please cite our paper:
```
@inproceedings{
  Raileanu2020RIDE:,
  title={{RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments}},
  author={Roberta Raileanu and Tim Rockt{\"{a}}schel},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rkg-TJBFPB}
}
```

## Installation

```
# create a new conda environment
conda create -n ride python=3.7
conda activate ride 

# install dependencies
git clone git@github.com:facebookresearch/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt

# install MiniGrid
cd gym-minigrid
python setup.py install
```

## Train RIDE on MiniGrid
```
cd impact-driven-exploration

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-ObstructedMaze-2Dlh-v0 

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-KeyCorridorS3R3-v0 \
--intrinsic_reward_coef 0.1 --entropy_cost 0.0005
```
To train RIDE on the other MiniGrid environments used in our paper, replace the ```--env``` argument above with each of the following:
```
MiniGrid-MultiRoom-N7-S4-v0
MiniGrid-MultiRoomNoisyTV-N7-S4-v0
MiniGrid-MultiRoom-N7-S8-v0
MiniGrid-MultiRoom-N10-S4-v0
MiniGrid-MultiRoom-N10-S10-v0
MiniGrid-MultiRoom-N12-S10-v0
MiniGrid-ObstructedMaze-2Dlh-v0 
MiniGrid-KeyCorridorS3R3-v0
```
Make sure to use the best hyperparameters for each environment, as listed in the paper. 

## Overview of RIDE
![RIDE Overview](/figures/ride_overview.png)

## Results on MiniGrid
![MiniGrid Results](/figures/ride_results.png)

## Analysis of RIDE
![Intrinsic Reward Heatmaps](/figures/ride_analysis.png)

![State Visitation Heatmaps](/figures/ride_analysis_counts.png)

## Acknowledgements
Our vanilla RL algorithm is based on [Torchbeast](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
