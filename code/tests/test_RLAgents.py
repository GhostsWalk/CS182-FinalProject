import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from train import read_command, train, run_games
from pacman import loadAgent


class TestRLAgents:
    def testExactQLearning(self):
        argv = ['--pacman=GreedyAgent', '--ghosts=ExactQLearningGhost',
                '--ghostArgs', 'partialObs=False', '-l', 'smallGrid',
                '--numTraining=500']
        args, options = read_command(argv)
        games = train(**args)
        scores = [-game.state.getScore() for game in games]

        ghosts = args["ghosts"]
        data = ghosts[0].export_data()
        for ghost in ghosts:
            ghost.load_from_data(data)
        num_runs = 10
        games = run_games(args, ghosts, num_runs=num_runs, graphics=False)
        wins = [game.state.getScore() < 0 for game in games]
        assert sum(wins) >= num_runs * 0.8

    def testExplorationExactQLearning(self):
        pass