import util
from util import manhattanDistance
from game import Actions


class FeatureExtractor:
    def get_features(self, state, action, **kwargs):
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def get_features(self, state, action, **kwargs):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class GhostFeatureExtractor(FeatureExtractor):
    def get_features(self, state, action, agent_index=None, **kwargs):
        """
        :param state: GameState
        :param action: Directions
        :param agent_index: int
        :param kwargs: others
        :return: features, util.Counter

        - distance to pacman, normalized by map size
        """
        features = util.Counter()

        x, y = state.getGhostPosition(agent_index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x+dx), int(y+dy)

        pacman_x, pacman_y = state.getPacmanPosition()
        current_distance = manhattanDistance((x, y), (pacman_x, pacman_y))
        future_distance = manhattanDistance((next_x, next_y), (pacman_x, pacman_y))

        if future_distance < current_distance:
            features["get_closer_to_pacman"] = 1
        else:
            features["get_closer_to_pacman"] = 0

        other_ghosts = state.getGhostPositions()
        other_ghosts = other_ghosts[:agent_index-1] + other_ghosts[agent_index:]
        curr_dists = [manhattanDistance((x, y), other) for other in other_ghosts]
        future_dists = [manhattanDistance((next_x, next_y), other) for other in other_ghosts]

        if len(other_ghosts) >= 1 and min(future_dists) < min(curr_dists):
            features["get_closer_to_other_ghosts"] = 1
        else:
            features["get_closer_to_other_ghosts"] = 0

        # walls = state.getWalls()
        # features['dist'] = float(dist) / (walls.width * walls.height)
        #
        # features['horizontal'] = next_x-pacman_x / (walls.width * walls.height)
        # features['vertical'] = next_y-pacman_y / (walls.width * walls.height)
        # features.divideAll(10.0)
        return features


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """
    def get_features(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
