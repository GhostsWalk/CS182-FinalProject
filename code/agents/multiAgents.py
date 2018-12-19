# multiAgents.py
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


import random, util
from game import Agent
from agent_utils.evaluationFunctions import *


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """\
        num_agents = gameState.getNumAgents()
        if num_agents < 2:
            raise Exception("Number of agents cannot be less than 2")

        # Compute score for each game state with minimax tree
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            raise Exception("Trying to find action when no action available")
        scores = []
        for index, action in enumerate(legal_actions):
            next_state = gameState.generateSuccessor(self.index, action)
            score = self.getScore(next_state, self.index+1, 1)
            scores.append(score)

        # Find best actions and return one of them randomly
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if
                        scores[index] == best_score]
        chosen_index = random.choice(best_indices)
        return legal_actions[chosen_index]

    def getScore(self, gameState, agentIndex, current_depth_level):
        """
        :param gameState: current game state in the search tree
        :param agentIndex: agent to act in the current state
        :param current_depth_level: current depth of the search tree
        :return: the minimax score at the current node
        """
        num_agents = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agentIndex)
        # If no action available, simply return current state value
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)

        # Simply return value at next states if at leaf of search tree
        if agentIndex == num_agents-1 and current_depth_level == self.depth:
            next_states = [gameState.generateSuccessor(agentIndex, action)
                           for action in legal_actions]
            scores = [self.evaluationFunction(state) for state in next_states]
            return min(scores)

        # Find scores of each action for current agent
        scores = []
        next_agent_index = (agentIndex + 1) % num_agents
        next_depth_level = current_depth_level + 1 if next_agent_index == 0 \
            else current_depth_level
        for index, action in enumerate(legal_actions):
            next_state = gameState.generateSuccessor(agentIndex, action)
            score = self.getScore(next_state, next_agent_index, next_depth_level)
            scores.append(score)

        # Find best score appropriate for the agent
        best_score = max(scores) if agentIndex == 0 else min(scores)
        return best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')
        beta = float('inf')
        action, _ = self.getBestActionScore(gameState, alpha, beta,
                                            self.index, 1)
        return action

    def getBestActionScore(self, gameState, alpha, beta, agentIndex,
                           current_depth):
        """
        :param gameState: game state to explore
        :param alpha: MAX best score seen so far
        :param beta: MIN best score seen so far
        :param agentIndex: index of operating agent
        :param current_depth: current depth level
        :return: best action and score for current agent at state
        """
        num_agents = gameState.getNumAgents()

        legal_actions = gameState.getLegalActions(agentIndex)
        # If no action available, simply return current state score
        if len(legal_actions) == 0:
            score = self.evaluationFunction(gameState)
            return None, score

        # If at end of depth, use evaluation function to make decision
        if agentIndex == num_agents-1 and current_depth == self.depth:
            best_score = float('inf')
            best_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agentIndex, action)
                score = self.evaluationFunction(next_state)
                if score < best_score:
                    best_score = score
                    best_action = action
                # If worse than what MAX could already achieve, no need to
                # explore further
                if score < alpha:
                    return action, score
            return best_action, best_score

        # Set initial score depending on agent being a MAX or MIN, and action
        best_score = float('-inf') if agentIndex == 0 else float('inf')
        best_action = None

        next_agent = (agentIndex+1)%num_agents
        next_depth = current_depth + 1 if next_agent == 0 else current_depth

        # alpha-beta pruning
        for action in legal_actions:
            next_state = gameState.generateSuccessor(agentIndex, action)
            _, score = self.getBestActionScore(next_state, alpha, beta,
                                               next_agent, next_depth)
            if (agentIndex == 0 and score > best_score) or \
                    (agentIndex != 0 and score < best_score):
                best_score = score
                best_action = action

            # If worse than what MIN or MAX could already achieve
            # respectively, no need to explore further
            if (agentIndex == 0 and score > beta) or (agentIndex != 0 and
                                                      score < alpha):
                return action, score

            # Book-keeping of current best
            if agentIndex == 0:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)

        return best_action, best_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        if num_agents < 2:
            raise Exception("Number of agents cannot be less than 2")

        # Compute score for each game state with expectimax tree
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            raise Exception("Trying to find action when no action available")
        scores = []
        for index, action in enumerate(legal_actions):
            next_state = gameState.generateSuccessor(self.index, action)
            score = self.getScore(next_state, self.index + 1, 1)
            scores.append(score)

        # Find best actions and return one of them randomly
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if
                        scores[index] == best_score]
        chosen_index = random.choice(best_indices)
        return legal_actions[chosen_index]

    def getScore(self, gameState, agentIndex, current_depth_level):
        """
        :param gameState: current game state in the search tree
        :param agentIndex: agent to act in the current state
        :param current_depth_level: current depth of the search tree
        :return: the expectimax score at the current node
        """
        num_agents = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agentIndex)
        # If no action available, simply return current state value
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)

        # Simply return value at next states if at leaf of search tree
        if agentIndex == num_agents-1 and current_depth_level == self.depth:
            next_states = [gameState.generateSuccessor(agentIndex, action)
                           for action in legal_actions]
            scores = [self.evaluationFunction(state) for state in next_states]
            return sum(scores)/float(len(scores))

        # Find scores of each action for current agent
        scores = []
        next_agent_index = (agentIndex + 1) % num_agents
        next_depth_level = current_depth_level + 1 if next_agent_index == 0 \
            else current_depth_level
        for index, action in enumerate(legal_actions):
            next_state = gameState.generateSuccessor(agentIndex, action)
            score = self.getScore(next_state, next_agent_index, next_depth_level)
            scores.append(score)

        # Find best score appropriate for the agent
        best_score = max(scores) if agentIndex == 0 else sum(scores)/float(len(
            scores))
        return best_score
