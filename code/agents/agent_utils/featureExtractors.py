import util
from util import manhattanDistance
from game import Actions


class FeatureExtractor:
    """ Abstract feature extractor
    """
    def get_features(self, state, action, **kwargs):
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    """ Identity Extractor that just returns the (state, action) pair
    """
    def get_features(self, state, action, **kwargs):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class GhostFeatureExtractor(FeatureExtractor):
    """ A more sophisticated feature extractor for ghosts
    """
    def get_features(self, state, action, agent_index=None, **kwargs):
        """
        :param state: GameState
        :param action: Directions
        :param agent_index: int
        :param kwargs: others
        :return: features, util.Counter
        """
        if state.__class__.__name__ == "Observation":
            # Use getting closer as a feature if partial observation
            pacman_x, pacman_y = state.pacman_absolute
            state = state.state

            features = util.Counter()
            x, y = state.getGhostPosition(agent_index)
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            current_distance = manhattanDistance((x, y), (pacman_x, pacman_y))
            future_distance = manhattanDistance((next_x, next_y),
                                                (pacman_x, pacman_y))

            if future_distance < current_distance:
                features["get_closer_to_pacman"] = 1.0
            else:
                features["get_closer_to_pacman"] = 0.0
            return features
        else:
            pacman_x, pacman_y = state.getPacmanPosition()
        features = util.Counter()
        features["bias"] = 1.0

        x, y = state.getGhostPosition(agent_index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x+dx), int(y+dy)

        # Get closer to pacman feature
        current_distance = manhattanDistance((x, y), (pacman_x, pacman_y))
        future_distance = manhattanDistance((next_x, next_y), (pacman_x, pacman_y))

        if future_distance < current_distance:
            features["get_closer_to_pacman"] = 1.0
        else:
            features["get_closer_to_pacman"] = 0.0

        # Get further from closest ghost feature
        other_ghosts = state.getGhostPositions()
        other_ghosts = other_ghosts[:agent_index-1] + other_ghosts[agent_index:]
        curr_dists = [manhattanDistance((x, y), other) for other in other_ghosts]
        future_dists = [manhattanDistance((next_x, next_y), other) for other in other_ghosts]

        if len(other_ghosts) >= 1 and min(future_dists) < min(curr_dists):
            features["get_closer_to_other_ghosts"] = 1.0
        else:
            features["get_closer_to_other_ghosts"] = 0.0

        # Count number of walls between self and pacman
        walls = state.getWalls()
        num_walls = 0
        inc_x = 1 if pacman_x > next_x else -1
        inc_y = 1 if pacman_y > next_y else -1
        for i in range(next_x+1, pacman_x, inc_x):
            for j in range(next_y+1, pacman_y, inc_y):
                if walls[i][j]:
                    num_walls += 1
        features["walls"] = num_walls
        features.divideAll(10.0)
        return features
