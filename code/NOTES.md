# Things to Take Notice of
## Time Penalty
How should time be taken into account for games?

For Pacman:
* when `win` in the end, longer time should reflect lower score
* when `lose` in the end, longer time should reflect higher score

For Ghosts:
* when `pacman die` (i.e. `lose` for pacman) in the end, longer time should reflect lower score
* when `pacman survive` (i.e. `win` for pacman) in the end, longer time should reflect higher score

Currently, no matter what, longer time gives lower score for **Pacman**, and higher score for **Ghosts**.

## Pacman Implementation
Currently, the **Pacman Agent** used for testing is `AlphaBetaAgent`. To better evaluate the performance of **Ghosts**, more intelligent **Pacman Agents** are needed.

## Storing Trained Agents
For ease of development and evaluation, a way to store and retrieve trained agents is needed.