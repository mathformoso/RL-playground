# RL-playground---ddpg
Playground for RL in Tensorflow

Own implementation of DDPG.

This version fits only continuous 1D space.
This version uses gym and Tensorflow1.8

Launch the script in command line with flag:
--train True for training
--env "env_name"for specifying gym environment
--path "path_name" for specifing the saving/loading model dir

Quick notes around ddpg:
DDPG takes advantage of Off-Policy learning for continuous control by alternatively:
- learning Q values with a parametrized function by minimizing TD error
- learning the actions maximizing these Q values with another parametrized function by gradient ascent over Q. 

A beautiful website @see: https://spinningup.openai.com/en/latest/algorithms/ddpg.html#
paper: https://arxiv.org/pdf/1509.02971.pdf
