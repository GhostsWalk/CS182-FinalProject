import random


class Observation:
    '''
    Class to model partial observation for agents (only used for ghosts)
    '''
    def __init__(self, state, agentIndex, maxDist=2):
        '''
        :param state: game state to be observed
        :param agentIndex: index of the observing agent
        :param maxDist: max dist of observation, i.e. the agent cannot see further than this distance
        '''
        self.maxDist = maxDist
        self.pos = state.getPacmanPosition() if agentIndex == 0 \
            else state.getGhostPosition(agentIndex)

        pacmanPos = state.getPacmanPosition()

        def noisy_relative_pacman(ghost_pos, pacman_pos):
            pacman = self.relativePos(ghost_pos, pacman_pos)
            p = random.random()
            if (p < 0.1):
                pacman = (pacman[0] + 1, pacman[1])
            elif (p < 0.2):
                pacman = (pacman[0] - 1, pacman[1])
            elif(p < 0.3):
                pacman = (pacman[0], pacman[1] + 1)
            elif(p < 0.4):
                pacman = (pacman[0], pacman[1] - 1)
            return pacman

        self.pacman = noisy_relative_pacman(self.pos, pacmanPos)

        self.ghosts = []
        self.ghosts_to_pacman = []
        for ind, other_pos in enumerate(state.getGhostPositions()):
            if ind != agentIndex-1:
                self.ghosts.append(self.relativePos(self.pos, other_pos))
                self.ghosts_to_pacman.append(noisy_relative_pacman(other_pos, pacmanPos))

        self.pacman_absolute = noisy_relative_pacman([0, 0], pacmanPos)

        # for pos in state.getGhostPositions():
        #     if self.withinDistance(pos):
        #         self.ghosts.add(self.relativePos(pos))



        # self.capsules = set()
        # for capsule in state.getCapsules():
        #     if self.withinDistance(capsule):
        #         self.capsules.add(self.relativePos(capsule))
        #
        # self.foods = set()
        # for food in state.getFood().asList():
        #     if self.withinDistance(food):
        #         self.foods.add(self.relativePos(food))

        self.state = state.deepCopy()
        self.index = agentIndex

    def withinDistance(self, pos2):
        dist = self.maxDist
        return abs(self.pos[0] - pos2[0]) <= dist and abs(self.pos[1] - pos2[1]) <= dist

    def relativePos(self, pos1, pos2):
        return pos2[0]-pos1[0], pos2[1]-pos1[1]

    ############################################
    # Functions for compatibility with dicts
    ############################################
    def __eq__(self, other):
        return self.pacman == other.pacman and self.ghosts == other.ghosts \
               and self.ghosts_to_pacman == other.ghosts_to_pacman

    def __hash__(self):
        return int((hash(tuple(self.ghosts)) + 7 * hash(self.pacman)) % 1048575)

    ############################################
    # Functions for retaining interfaces when
    # treating observations as states
    ############################################
    def getLegalActions(self, agentIndex):
        return self.state.getLegalActions(agentIndex=agentIndex)

    def getScore(self):
        return self.state.getScore()

    ############################################
    # Functions for ease of representation
    # especially when debugging
    ############################################
    def __str__(self):
        out = ["agent #{}".format(self.index),
               "pos: {}".format(self.pos),
               "ghosts: {}".format(self.ghosts),
               "pacman: {}".format(self.pacman),
               "ghosts_to_pacman: {}".format(self.ghosts_to_pacman)]
        return "\n".join(out)

    def __repr__(self):
        return str(self)

