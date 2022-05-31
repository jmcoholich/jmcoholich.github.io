---
title: A Bag of Tricks for Reinforcement Learning
subtitle:
# Summary for listings and search engines
summary: Google Foobar

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
  caption: 'Completion screen'
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

I first started studying reinforcement learning (RL) in Summer 2020, when I started
working under Professor [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/). The results in the field of RL were amazing, but it turned out to be much more difficult than expected to get RL algorithms to work on new problems or to implement RL algorithms from scratch (the best way to learn in my past experience). These [blog](https://www.alexirpan.com/2018/02/14/rl-hard.html#:~:text=Often%2C%20it%20doesn't%2C,out%20of%20the%20RL%20algorithm.) [posts](https://andyljones.com/posts/rl-debugging.html) do a great job of explaining the difficulty of getting into RL, which really resonate with my own experiences.

The purpose of this blog post is to enumerate the small tricks that I have discoverd during the past two years which make RL work or work better. They are roughly ordered in descencing order of importance. Where possible, I have tried to include links to RL implementations where these tricks are implemented.

My issue was that even if I understood all the theory behind an RL algorithm, either from a paper or from Spinning Up, there were still many tricks required to get RL working in practice. To draw a parallel to supervised learning, it would be like understanding SGD and neural networks, but not having knowledge of batch norms or residual connections.

I hope this is helpful to someone who is starting out. Let me know if you think there is anything that needs to be added to this list!

### gradient normalization and clipping
### advantage standardization
### observation normalization and clipping
### reward normalization and clipping
### A dense and smooth reward function (dense = every timestep, smooth = varies smoothly between regions of the state space (ie gradual change vs large steps))
### value function update clipping
### PPO loss
### shared actor-critic layers
### entropy loss decay
### adaptive learning rate based on KL-div
### ADAM optimizer
### hyperparameter tuning
### bootstrapping good terminations
### Generalized Advantage Estimation

