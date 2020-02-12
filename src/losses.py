# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np 


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(torch.mean(advantages**2, dim=1))


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction='none')
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))


def compute_forward_dynamics_loss(pred_next_emb, next_emb):
    forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)
    return torch.sum(torch.mean(forward_dynamics_loss, dim=1))


def compute_inverse_dynamics_loss(pred_actions, true_actions):
    inverse_dynamics_loss = F.nll_loss(
        F.log_softmax(torch.flatten(pred_actions, 0, 1), dim=-1), 
        target=torch.flatten(true_actions, 0, 1), 
        reduction='none')
    inverse_dynamics_loss = inverse_dynamics_loss.view_as(true_actions)
    return torch.sum(torch.mean(inverse_dynamics_loss, dim=1))


