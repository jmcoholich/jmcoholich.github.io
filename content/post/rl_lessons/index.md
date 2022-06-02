---
title: A Bag of Tricks for Deep Reinforcement Learning
subtitle:
# Summary for listings and search engines
summary:

# Link this post with a project
projects: []

# Date published
date: '2022-05-31T00:00:00Z'

# Date updated
lastmod: '2022-05-31T00:00:00Z'

# Is this an unpublished draft?
draft: true

# Show this page in the Featured widget?
featured: false

share: False

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'The KUKA bin environment visualized in NVIDIA IsaacGym'
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - admin

# tags:
#   - Academic

# categories:
#   - Demo
#   - 教程
---

It is difficult to reproduce results in deep reinforcement learning ([Wired article](https://www.wired.com/story/artificial-intelligence-confronts-reproducibility-crisis/)). I first started studying reinforcement learning (RL) in the summer of 2020, when I joined the [Robotics Perception and Learning Lab](https://faculty.cc.gatech.edu/~zk15/). I had been motivated by the amazing results achieved so far in board games, video games, and robotics and the promise of how powerful and general the methods are. While its trivial to clone a popular RL repository and run trianing on RL benchmarks, I found that on new problems or to implement RL algorithms from scratch is quite difficult, especially for a beginner(the best way to learn in my past experience).

Understand the theory and academic papers behind an RL algorithm is essential to doing research, but not enough practically.

My issue was that even if I understood all the theory behind an RL algorithm, either from a paper or from Spinning Up, there were still many tricks required to get RL working in practice. To draw a parallel to supervised learning, it would be like understanding SGD and neural networks, but not having knowledge of batch norms or residual connections.

Altough PPO is a SOTA algorithm, implementing pseudocode directly from the PPO paper (below) will not yeild SOTA performance. You need all the other stuff.


This seems to be a somewhat common occurence in the field of RL, based off of my conversations with other students at Georgia Tech. [These](https://www.alexirpan.com/2018/02/14/rl-hard.html#:~:text=Often%2C%20it%20doesn't%2C,out%20of%20the%20RL%20algorithm.) blog [posts](https://andyljones.com/posts/rl-debugging.html) do a great job of explaining this and really resonate with my own experiences.

The purpose of this post is to enumerate the small tricks that I have discoverd during the past two years which make RL work or work better. Some of these "tricks" are will be obvious if you have experience in supervised deep learning, such as gradient clipping and input normalization. These are all things that are very important to getting this working, but are mundane enough that most of the time no one really tells you explicitly to make sure to do these things or they assume that you already know. They are roughly ordered in descending order of importance. Where possible, I have tried to include links to code in RL implementations where these tricks are found. I will additionally include a link to any help Pytorch functions for implementation.

Some of these things you really only need to worry about if you decide to write an RL algorithm from scratch, as most implementations will already include them.



I hope this is helpful to someone who is starting out. Please don't hesitate to reach out to me if you think there is something missing from here!


Most of the example will come from either of these two RL implementations of which I am familair with:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
https://github.com/Denys88/rl_games


For reference, nearly all of my work in RL has been using proximal policy optimization (PPO) for continuous control.




TODO post this on the RL discord.

I don't have ablation results on all of these.

### Using an existing RL Implementation and environment
This is the main thing you should do instead of trying to code one from scratch. Take an existing implementation, play around with it, run some benchmarks. Then make a fork and start modifying the implementation for your own project. They will include their own set of tricks, and the creators have likely already tuned it a lot on RL benchmarks and provide default hyperparameter values that work decently well.

### Observation Normalization and Clipping

This was a big one, and should be somewhat obvious if you have a background in supervised learning.

Neural networks like nice smooth inputs and outputs.

The differnece in RL is that (unless you are doing offline RL or behavioral cloning) you don't have the entire dataset upfront to calculate statistics with. Your dataset is generated online through interactions with the envirionment as training progresses, so your statistics will change online as well. Pretty much every implemtation uses a "running_mean_std" class like [this](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py) from Open AI. which uses [Welford's Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).


Using an online mean and std typically causes an initial drop in performance as the mean and std initially move very quickly in the beginning due to low "count" values and exploration (in the beginning of training, the agent ecounters new areas of the state space rapidly.) (show pic)

![pic](obs_norm_dip.png)


https://github.com/Denys88/rl_games/blob/06a3319d3a6af566d984aa5953b1fd7a24a8e3a4/rl_games/common/a2c_common.py#L587

https://github.com/Denys88/rl_games/blob/94e55563be60f10e659428cdce7b4e0bd131d471/rl_games/algos_torch/models.py#L41

Note: One common bug is the failure to save and load normailzation statistics in addition to model files. An RL policy will only achieve the training reward when the inputs go through the same norm stats as during training.



### A dense and smooth reward function

(dense = every timestep, smooth = varies smoothly between regions of the state space (ie gradual change vs large steps))

Sparse rewards are difficult for RL algorithms to learn from, and an ongoing subject of research. If possible, try densifying your rewards.

Give footstep example from my own work and from ALLSTEPS.



### Gradient Normalization and Clipping
https://github.com/Denys88/rl_games/blob/8da6852f72bdbe867bf12f792b00df944b419c43/rl_games/common/a2c_common.py#L252


[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html)


### Advantage Standardization
What is the intuition behind this one again?

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/algo/ppo.py#L36



### reward normalization and clipping
Another big one.




### value function update clipping
https://github.com/Denys88/rl_games/blob/8da6852f72bdbe867bf12f792b00df944b419c43/rl_games/common/common_losses.py#L7
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/algo/ppo.py#L68


### PPO loss
### shared actor-critic layers


### Learning Rate Scheduling
A commong thing is to implement a linearly decreasing learning rate throughout training. The idea is that towards the end of training, you want to avoid making descrutively large policy updates (this is also the idea behind TRPO and allocating your retirement savings into bonds as you grow older.) and your performance will have mostly saturated so you should just be fine-tuning your policy.

You could also be fancy and do this lr based which adapts based on a desired [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between something. I have mostly used this for my work with the quadruped, set the initial learning rate to 1e-5 and then let the adaptive lr take over. Here is what the lr plot usually looks like.
![pic](adaptive_lr.png)

This shows five training runs for my [quadruped project](https://www.jeremiahcoholich.com/publication/quadruped_footsteps/). The desired KL-divergence was set to 0.01. The lr usually hovers around 7e-4.

Manually schedule and set lr:
https://github.com/Denys88/rl_games/blob/50d9a460f8ba41de5dbac4abed04f8de9b849f4f/rl_games/common/schedulers.py#L19
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/efc71f600a2dca38e188f18ca85b654b37efd9d2/a2c_ppo_acktr/utils.py#L46

To step lr from within pytorch:
https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html




### ADAM optimizer
### hyperparameter tuning
RL is notoriously sensitive to hyperparameters and there is no one-size-fits all for good hyperparameter values. Typically, different implementations and different applications will need different hyperparameters. This is a major complaint about RL.

The good thing is, [Weights & Biases](https://wandb.ai/site) has a very good pipeline for doing automated and disstributed hyperparm sweeps.
[Check it out.](https://docs.wandb.ai/guides/sweeps)

### bootstrapping good terminations
I have found that this is not stricly necessary (afaik, the rl_games library does without it) and can sometimes hurt (excess bootstrapping can sometimes hurt, which is a hypothesized reason that DQN and td-learning doesn't do that well).

This idea is this

{{< math >}}$$ e^{i \pi}$${{< /math >}}

### Generalized Advantage Estimation
I have found GAE is useful in improving performance. Theroetically, given enough samples, you can do without it. Its just another knob to turn. In most training runs, I set gamma to 0.99 and lambda = 0.95.

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/efc71f600a2dca38e188f18ca85b654b37efd9d2/a2c_ppo_acktr/storage.py#L73

### entropy loss decay
The idea is simple: in the beginning you want more exploration. Towards the end you want more exploitation. This can help your agent avoid local optima early in training. I have often found this is uncessary, and don't really use it.



