import random
from pacman import default, get_default_parser
import sys
from train import load_setup, run_games, load_ghosts


def read_command(argv):
    parser = get_default_parser()
    parser.add_option('-f', '--filename', dest='filename', type='str',
                      help=default(
                          'file to load ghost from'))

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = dict()

    random.seed('cs188')

    layout, pacman, ghosts = load_setup(options)
    args['layout'] = layout
    args['pacman'] = pacman
    args['ghosts'] = ghosts
    args['filename'] = options.filename

    return args, options


if __name__ == "__main__":
    args, options = read_command(sys.argv[1:])
    ghosts = load_ghosts(options, options.filename)
    run_games(args, ghosts)
