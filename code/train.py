import random
import layout
from pacman import parseAgentArgs, loadAgent, ClassicGameRules, get_default_parser
import textDisplay
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_setup(options):
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


def load_ghosts(options):
    ghostType = loadAgent(options.ghost, nographics=False)
    ghostOpts = parseAgentArgs(options.ghostArgs)
    ghosts = [
        ghostType(index=i + 1, numTraining=options.numTraining, **ghostOpts)
        for i in range(options.numGhosts)]
    for ghost in ghosts:
        ghost.load_from_file(filename)
        ghost.turnoff_training()
    return ghosts


def run_games(args, ghosts, num_runs=5, graphics=True):
    if graphics:
        import graphicsDisplay
        display = graphicsDisplay.PacmanGraphics(frameTime=0.1)
    else:
        import textDisplay
        display = textDisplay.NullGraphics()
    import __main__
    __main__.__dict__['_display'] = display

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

    # Compute average score of last 10 episodes for smoothness in graph
    # averages = [np.mean(scores[i-10:i+1]) for i in range(len(scores))]
    # Compute running averages for smoothness in graph
    averages = [np.mean(scores[:i + 1]) for i in range(len(scores))]

    ghost_class_name = args['ghosts'][0].__class__.__name__
    plt.plot(averages)
    plt.title(ghost_class_name)
    plt.xlabel("training episodes")
    plt.ylabel("running average scores")
    figname = "plots/{}-{}-{}".format(ghost_class_name, options.layout,
                                      args["numTraining"])
    plt.savefig(figname)
    print("Plot successfully saved to {}".format(figname))
    plt.show()

    ans = raw_input("Save the agent? (y/n) ")
    if ans == "y":
        ghost = args["ghosts"][0]
        filename = "trained/{}-{}-{}-partialObs_{}.pkl".format(ghost_class_name, options.layout,
                               args["numTraining"], ghost.partialObs)
        ghost.save_to_file(filename)
        print("Agent successfully saved to {}".format(filename))

        ghosts = load_ghosts(options)
        games = run_games(args, ghosts)
