# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
from agent_util import scoreEvaluationFunction
from agent_util import evaluationFunctionWithDistance


class GhostAgent( Agent ):
    def __init__( self, index):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist


class AlphaBetaGhost(GhostAgent):
    def __init__(self, index, evalFn='evaluationFunctionWithDistance', depth='2', **kwargs):
        self.index = index
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.max_depth = int(depth)

    def getAction( self, state ):
        alpha = float("-inf")
        beta = float("inf")
        action, _ = self.getBestActionScore(state, alpha, beta,
                                            self.index, 1)
        return action

    def getBestActionScore(self, state, alpha, beta, agent_index, current_depth):
        legal_actions = state.getLegalActions(agent_index)
        if len(legal_actions) == 0:
            score = self.evaluationFunction(state)
            return None, score

        num_agents = state.getNumAgents()
        if current_depth == self.max_depth and self.endOfPly(agent_index, num_agents):
            return self.evalBestActionScore(agent_index, state, alpha, beta)


        # Initialization depending on node type (MAX or MIN)
        best_score = float("-inf") if agent_index == 0 else float("inf")
        best_action = None
        next_agent = (agent_index+1)%num_agents
        next_depth = current_depth + 1 if next_agent == 0 else current_depth

        # Alpha-Beta Pruning
        for action in legal_actions:
            next_state = state.generateSuccessor(agent_index, action)
            _, score = self.getBestActionScore(next_state, alpha, beta, next_agent, next_depth)
            if (agent_index == 0 and score >= beta) or\
                    (agent_index != 0 and score <= alpha):
                return action, score

            if (agent_index == 0 and score > best_score) or \
                    (agent_index != 0 and score < best_score):
                best_score = score
                best_action = action

            if agent_index == 0:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
        return best_action, best_score

    def evalBestActionScore(self, agent_index, state, alpha, beta):
        """
        :param agent_index: int
        :param state: GameState
        :param alpha: float
            From this node back to root, the value a MAX node can already achieve
        :param beta: float
            From this node back to root, the value a MIN node can already achieve
        :return: the best action for agent of agent_index to take, using evaluation func
        """
        legal_actions = state.getLegalActions(agent_index)

        if agent_index != 0:  # Ghost, so aim to minimize score
            best_score = float("inf")
            best_action = None
            for action in legal_actions:
                next_state = state.generateSuccessor(agent_index, action)
                score = self.evaluationFunction(next_state)
                if score < best_score:
                    best_score = score
                    best_action = action
                if score <= alpha:
                    # A score worse than MAX node could achieve,
                    # So a MAX node would not let this happen
                    return action, score
        else:
            best_score = float("-inf")
            best_action = None
            for action in legal_actions:
                next_state = state.generateSuccessor(agent_index, action)
                score = self.evaluationFunction(next_state)
                if score > best_score:
                    best_score = score
                    best_action = action
                if score >= beta:
                    # A score worse than MIN node could achieve,
                    # So a MIN node would not let this happen
                    return action, score

        return best_action, best_score

    @staticmethod
    def endOfPly(current_agent, num_agents):
        """
        :param current_agent: int
        :param num_agents: int
        :return: True if end of a ply (a run for each agent)
        """
        return current_agent == num_agents-1
