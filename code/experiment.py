import sys

import matplotlib.pyplot as plt
import numpy as np

import layout
import textDisplay
from pacman import loadAgent, ClassicGameRules, parseAgentArgs, default


def read_command(argv):
    from optparse import OptionParser
    usage_str = """
    USAGE:      python experiment.py <options>
    EXAMPLES:   (1) python experiment.py --layout testRL --pacman=AlphaBetaAgent --ghosts=QLearningGhost --numTraining=10 
    """
    parser = OptionParser(usage_str)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help = default('the agent TYPE in the pacmanAgents module to use'),
                      metavar = 'TYPE', default = 'KeyboardAgent')
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar='TYPE', default='QLearningGhost')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'),
                      default=10)
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'),
                      default=4)
    parser.add_option('-f', '--fixRandomSeed', action='store_true',
                      dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game',
                      default=False)
    parser.add_option('--ghostArgs', dest='ghostArgs',
                      help='Comma separated values sent to ghost. e.g. "opt1=val1,opt2,opt3=val3"')

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = dict()

    # Fix the random seed
    if options.fixRandomSeed: random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] is None: raise Exception(
        "The layout " + options.layout + " cannot be found")

    # Initialize a Pacman agent
    pacmanType = loadAgent(options.pacman, nographics=False)
    pacmanOpts = parseAgentArgs(options.agentArgs)
    pacman = pacmanType(**pacmanOpts)  # Instantiate Pacman with depth=2 # To Do: allow depth as an argument
    args['pacman'] = pacman

    # Initialize Ghost agents
    ghostType = loadAgent(options.ghost, nographics=False)
    ghostOpts = parseAgentArgs(options.ghostArgs)
    args['ghosts'] = [ghostType(index=i + 1, numTraining=options.numTraining, **ghostOpts) for i in range(options.numGhosts)]

    # Set number of training episodes for experiment
    args['numTraining'] = options.numTraining

    return args, options


def run_experiment(layout, pacman, ghosts, numTraining, catchExceotions=False, timeout=30):
    """ Run an experiment with the provided layout, pacman and ghosts
        Returns the game after it is run
    """
    rules = ClassicGameRules(timeout)
    games = []
    for i in range(numTraining):
        game_display = textDisplay.NullGraphics()
        rules.quiet = True
        game = rules.newGame(layout, pacman, ghosts, game_display, quiet=True,
                             catchExceptions=catchExceotions)
        game.run()
        games.append(game)

    return games


if __name__ == "__main__":
    args, options = read_command(sys.argv[1:])
    games = run_experiment(**args)
    scores = [-game.state.getScore() for game in games]

    # Compute running averages for smoothness in graph
    averages = [np.mean(scores[:i + 1]) for i in range(len(scores))]

    # Plot and save the training results
    ghost_class_name = args['ghosts'][0].__class__.__name__
    plt.plot(averages)
    plt.title(ghost_class_name)
    plt.xlabel("training episodes")
    plt.ylabel("running average scores")
    plt.savefig("plots/{}-{}-{}".format(ghost_class_name, options.layout,
                               args["numTraining"]))
    plt.show()
