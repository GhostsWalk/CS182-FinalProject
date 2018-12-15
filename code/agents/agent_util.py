from util import manhattanDistance
import util


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


def evaluationFunctionWithDistance(state):
    """
    :param state: GameState
    :return: evaluate score + heuristic that includes distance
    """
    score = state.getScore()
    pacman_pos = state.getPacmanPosition()
    positions = state.getGhostPositions()
    distances = [manhattanDistance(pacman_pos, ghost) for ghost in positions]
    return score + sum(distances)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
