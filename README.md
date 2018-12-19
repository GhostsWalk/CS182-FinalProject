# CS182 Final Project

Team Members: Summer Yuan, Jay Li

This project investigates the problem of reinforcement learning for partially observable Markov decision process (POMDP) with multiple agents.

To study this problem, we used the open-source Pac-Man environment, as the basis environment to develop our algorithms and experiments. To model our reinforcement learning problem, we take the standpoint of the ghosts, the goal of which is to capture the pacman as fast as possible. It is assumed that all the ghosts share the common rewards. Otherwise, the problem would become stochastic game which is outside the domain of this project. The pacman agent is set to be a random agent so that it becomes part of the environment, and contributes to the probabilistic transition model. If the pacman has a goal, such as the case when it is a Minimax agent, the problem would again become a stochastic game, that falls outside the scope of this project. To model the partial observation, we cut down the original full state information into a noisy relative position measurement between the ghosts and pacman.
