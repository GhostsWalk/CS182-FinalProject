import random

class Observation:
    '''
    Class to model partial observation for agents
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
        self.pacman = self.relativePos(pacmanPos) # if self.withinDistance(pacmanPos) else None
        if self.pacman != None:
            p = random.random()
            if (p < 0.1):
                self.pacman = (self.pacman[0] + 1, self.pacman[1])
            elif (p < 0.2):
                self.pacman = (self.pacman[0] - 1, self.pacman[1])
            elif(p < 0.3):
                self.pacman = (self.pacman[0], self.pacman[1] + 1)
            elif(p < 0.4):
                self.pacman = (self.pacman[0], self.pacman[1] - 1)
        self.ghosts = set()
        for pos in state.getGhostPositions():
            if self.withinDistance(pos):
                self.ghosts.add(self.relativePos(pos))

        self.capsules = set()
        for capsule in state.getCapsules():
            if self.withinDistance(capsule):
                self.capsules.add(self.relativePos(capsule))

        self.foods = set()
        for food in state.getFood().asList():
            if self.withinDistance(food):
                self.foods.add(self.relativePos(food))

        self.state = state.deepCopy()
        self.index = agentIndex

    def withinDistance(self, pos2):
        dist = self.maxDist
        return abs(self.pos[0] - pos2[0]) <= dist and abs(self.pos[1] - pos2[1]) <= dist

    def relativePos(self, pos2):
        return pos2[0]-self.pos[0], pos2[1]-self.pos[1]

    ############################################
    # Functions for compatibility with dicts
    ############################################
    def __eq__(self, other):
        return self.pacman == other.pacman and self.ghosts == other.ghosts and \
                self.capsules == other.capsules and self.foods == other.foods

    def __hash__(self):
        return int((hash(tuple(self.ghosts)) + 13 * hash(tuple(
            self.foods)) + 113 * hash(tuple(self.capsules)) + 7 * hash(
            self.pacman)) % 1048575)

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
               "foods: {}".format(self.foods),
               "capsules: {}".format(self.capsules)]
        return "\n".join(out)

    def __repr__(self):
        return str(self)
