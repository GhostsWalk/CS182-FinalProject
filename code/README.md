# How to Run the Pacman Code

### Default Play
`python pacman.py`

This will run pacman with keyboard control and random ghost agents.

### Play w/ Specified Pacman Agent and Ghost Agent

eg.
`python pacman.py --pacman=GreedyAgent --ghosts=DirectionalGhost`

where `GreedyAgent` is an `Agent` Class in [pacmanAgents.py](./pacmanAgents.py) and `DirectionalGhost` is a `GhostAgent` Class in [ghostAgents.py](./ghostAgents.py).

Any file with postfix `gents.py` will searched when looking for agents.


### Test independently learned RL Ghost Agents
eg. `python pacman.py --pacman=AlphaBetaAgent --ghosts=QLearningGhost --numTraining=100 -l testRL -n 110`

The last (110 - 100) games will be displayed


### To Enable Partial Observation
eg. `python pacman.py --pacman=AlphaBetaAgent --ghosts=QLearningGhost --numTraining=100 -l testRL -n 110 --ghostArgs partialObs=True`

The default partial observation scheme is that the agents can not see further than 2 squares away from it in any directions.


# How to Run Experiment

eg. `python experiment.py --layout testRL --pacman=AlphaBetaAgent --ghosts=QLearningGhost --numTraining=500`

It will generate a plot of the experiment, and save to `plots/{ghostType}-{numTraining}.png`.