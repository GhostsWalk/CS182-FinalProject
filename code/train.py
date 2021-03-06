import random
import layout
from pacman import parseAgentArgs, loadAgent, ClassicGameRules, get_default_parser
import textDisplay
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_setup(options):
    """ Load the setup for training the agent
    """
    layout_ = layout.getLayout(options.layout)
    if layout is None:
        raise Exception("The layout " + options.layout + " cannot be found")

    # Initialize a Pacman agent
    pacmanType = loadAgent(options.pacman, nographics=False)
    pacmanOpts = parseAgentArgs(options.agentArgs)
    pacman = pacmanType(
        **pacmanOpts)  # Instantiate Pacman with depth=2 # To Do: allow depth as an argument

    # Initialize Ghost agents
    ghostType = loadAgent(options.ghost, nographics=False)
    ghostOpts = parseAgentArgs(options.ghostArgs)
    ghosts = [
        ghostType(index=i + 1, numTraining=options.numTraining, **ghostOpts)
        for i in range(options.numGhosts)]

    return layout_, pacman, ghosts


def read_command(argv):
    """ Read command lines arguments and return setups for the game
    """
    parser = get_default_parser()

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = dict()

    random.seed('cs188')

    layout_, pacman, ghosts = load_setup(options)
    args['layout'] = layout_
    args['pacman'] = pacman
    args['ghosts'] = ghosts

    # Set number of training episodes for experiment
    args['numTraining'] = options.numTraining

    return args, options


def train(layout, pacman, ghosts, numTraining, catchExceotions=False, timeout=30):
    """ Train the ghost with the specified game params
    """
    rules = ClassicGameRules(timeout)
    games = []
    # Run the game with numTraining times
    for i in range(numTraining):
        game_display = textDisplay.NullGraphics()
        rules.quiet = True
        game = rules.newGame(layout, pacman, ghosts, game_display, quiet=True,
                             catchExceptions=catchExceotions)
        game.run()
        games.append(game)

    return games


def load_ghosts(options, filename):
    """ Load the ghosts from file
    :param options: options from commandline
    :param filename: pkl file that stores the agent
    :return: ghosts set from pkl file
    """
    # Initialize ghosts given type
    ghostType = loadAgent(options.ghost, nographics=False)
    ghostOpts = parseAgentArgs(options.ghostArgs)
    ghosts = [
        ghostType(index=i + 1, numTraining=options.numTraining, **ghostOpts)
        for i in range(options.numGhosts)]
    # Load ghosts from file and turn off training
    for ghost in ghosts:
        ghost.turnoff_training()
        ghost.load_from_file(filename)
    return ghosts


def run_games(args, ghosts, num_runs=5, graphics=True):
    """ Run the games with the setup
    """
    if graphics:
        import graphicsDisplay
        display = graphicsDisplay.PacmanGraphics(frameTime=0.1)
    else:
        import textDisplay
        display = textDisplay.NullGraphics()
    import __main__
    __main__.__dict__['_display'] = display

    # Run games with classic rules
    rules = ClassicGameRules(timeout=30)
    games = []
    for _ in range(num_runs):
        game = rules.newGame(args['layout'], args['pacman'], ghosts,
                             display)
        game.run()
        games.append(game)
    return games


if __name__ == '__main__':
    args, options = read_command(sys.argv[1:])
    games = train(**args)
    scores = [-game.state.getScore() for game in games]

    # Compute running averages for smoothness in graph
    averages = [np.mean(scores[:i + 1]) for i in range(len(scores))]

    # Plot the training results and save them
    ghost = args['ghosts'][0]
    ghost_class_name = ghost.__class__.__name__
    plt.plot(averages)
    plt.title(ghost_class_name)
    plt.xlabel("training episodes")
    plt.ylabel("running average scores")
    figname = "plots/{}-{}-{}-partialObs_{}".format(ghost_class_name, options.layout,
                                      args["numTraining"], ghost.partialObs)
    plt.savefig(figname)
    print("Plot successfully saved to {}".format(figname))
    plt.show()

    # Save the agent and let them play games with graphics on
    ans = raw_input("Save the agent? (y/n) ")
    if ans == "y":
        ghost = args["ghosts"][0]
        filename = "trained/{}-{}-{}-partialObs_{}.pkl".format(ghost_class_name, options.layout,
                               args["numTraining"], ghost.partialObs)
        ghost.save_to_file(filename)
        print("Agent successfully saved to {}".format(filename))

        ghosts = load_ghosts(options, filename)
        for ghost in ghosts:
            if hasattr(ghost, 'getWeights'):
                print(ghost.getWeights())
        games = run_games(args, ghosts)
