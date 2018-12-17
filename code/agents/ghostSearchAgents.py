from ghostAgents import GhostAgent
from agent_utils.evaluationFunctions import *
import util


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


class Node:
    def __init__(self, state, action, parent):
        self.state = state
        self.action = action
        self.parent = parent

    def __eq__(self, other):
        return self.state == other.state

    def first_action(self):
        node = self
        while node.parent is not None:
            node = node.parent
        return node.action


class BFSGhost(GhostAgent):
    def __init__(self, index, **kwargs):
        self.index = index

    def getAction(self, state):
        pacman = state.getPacmanPosition()
        from collections import deque
        queue = deque()

        explored = set()
        legal_actions = state.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return None
        else:
            for action in legal_actions:
                next_state = state.generateSuccessor(self.index, action)
                node = Node(next_state, action, None)
                queue.append(node)
                explored.add(next_state)
            while len(queue) > 0:
                node = queue.popleft()
                state = node.state

                self_pos = state.getGhostPosition(self.index)
                if self_pos == pacman:
                    return node.first_action()

                legal_actions = state.getLegalActions(self.index)
                for action in legal_actions:
                    next_state = state.generateSuccessor(self.index, action)
                    if next_state not in explored:
                        explored.add(next_state)
                        next_node = Node(next_state, action, node)
                        queue.append(next_node)
        return None
