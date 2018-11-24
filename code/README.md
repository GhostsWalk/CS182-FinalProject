# How to Run the Pacman Code

### Default Play
`python pacman.py`

This will run pacman with keyboard control and random ghost agents.

### Play w/ Specified Pacman Agent and Ghost Agent

eg.
`python pacman.py --pacman=GreedyAgent --ghost=DirectionalGhost`

where `GreedyAgent` is an `Agent` Class in [pacmanAgents.py](./pacmanAgents.py) and `DirectionalGhost` is a `GhostAgent` Class in [ghostAgents.py](./ghostAgents.py).