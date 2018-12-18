import time, random
from ghostAgents import GhostAgent
from agent_utils.featureExtractors import *
import util
import cPickle as pickle
from util import string_to_bool

shared_q = util.Counter()
shared_weights = util.Counter()


class AbstractQLearningGhost(GhostAgent):
    def __init__(self, index, actionFn=None, numTraining=100, alpha=0.5, gamma=0.9, partialObs=True, epsilon=0.9, exploration="False", shareQ=False):
        """
        :param index: int, agent index
        :param actionFn: func, function to get list of legal actions
        :param numTraining: int, number of training episodes
        :param alpha: float, learning rate
        :param gamma: float, discount factor
        :param partialObs: bool, whether the agent takes partial observation
        """
        self.index = index
        if actionFn is None:
            actionFn = lambda state: state.getLegalActions(agentIndex=index)
        self.actionFn = actionFn
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.partialObs = string_to_bool(partialObs)
        self.shareQ = string_to_bool(shareQ)
        if self.shareQ:
            global shared_q
            shared_q = util.Counter()
            global shared_weights
            shared_weights = util.Counter()

        self.epsilon = float(epsilon)
        self.exploration = string_to_bool(exploration)
        if self.epsilon == 0 and not self.exploration:
            raise Exception("Need either epsilon-greedy or exploration function for exploration")
        if self.epsilon != 0 and self.exploration:
            raise Exception("Cannot have both epsilon-greedy and exploration function for exploration")
        self.N = util.Counter()

        self.q_values = util.Counter()  # Map (state, action) to value

        # Episode records
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0
        self.episodesSoFar = 0

    def save_to_file(self, filename):
        data = self.export_data()
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.load_from_data(data)

    def load_from_data(self, data):
        for key, val in data.items():
            setattr(self, key, val)

    def computeActionFromQValues(self, state):
        """
        :param state: GameState
        :return: Directions, action that the agent would take
        """
        legal_actions = self.actionFn(state)
        if len(legal_actions) == 0:
            return None
        best_action = None
        best_value = None
        for action in legal_actions:
            value = self.getQValue(state, action)
            if best_value is None or value > best_value:
                best_action = action
                best_value = value
        return best_action

    def getValue(self, state):
        """ Get value of state, by taking max of q_values over actions
        :param state: GameState
        :return: numeric
        """
        legal_actions = self.actionFn(state)
        if len(legal_actions) == 0:
            return 0
        best_value = None
        for action in legal_actions:
            value = self.getQValue(state, action)
            if self.exploration:
                K = float(1000)
                value += K/(self.N[(state, action)] + 1)
            if best_value is None or value > best_value:
                best_value = value
        return best_value

    def observeTransition(self, state, action, next_state, reward):
        """ Observe a sample of state, action -> new_state transition, to update q value
        :param state: GameState
        :param action: Directions
        :param next_state: GameState
        :param reward: numeric
        :return:
        """
        self.episodeRewards += reward
        self.updateQValue(state, action, next_state, reward)
        self.N[(state, action)] += 1

    def observationFunction(self, state):
        """
           This is where we ended up after our last action.
           The simulation should somehow ensure this is called
        """

        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        if self.lastState is not None:
            reward = self.getReward(state)
            self.observeTransition(self.lastState, self.lastAction, obs, reward)
        return obs

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print("Beginning %d episodes of Training" % self.numTraining)

    def final(self, state):
        """ Called by Pacman game at the terminal state
        :param state: GameState
        """
        self.observationFunction(state)
        self.stopEpisode()

        if 'episodeStartTime' not in self.__dict__:
            self.episodeStartTime = time.time()
        if 'lastWindowAccumRewards' not in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += -state.getScore()  # Total reward of an episode

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status for agent {}: '.format(self.index))
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                # Report training
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print("\tCompleted %d out of %d training episodes" % (
                    self.episodesSoFar, self.numTraining))
                print("\tAverage Rewards over all training %.2f" % trainAvg)
            else:
                # Report testing
                test_episodes = self.episodesSoFar-self.numTraining
                testAvg = float(self.accumTestRewards) / test_episodes
                print("\tCompleted %d test episodes" % test_episodes)
                print("\tAverage Rewards over testing: %.2f" % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, windowAvg))
            print('\tEpisode took %.2f seconds' % (
                    time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = "Training Done"
            print('%s\n%s' % (msg,'-' * len(msg)))

    def getReward(self, state):
        """ Get reward for ghost agent
        :param state: GameState
        :return: reward, numeric
        """
        return -(state.getScore() - self.lastState.getScore())

    def startEpisode(self):
        """
            Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            # Still in training
            self.accumTrainRewards += self.episodeRewards
        else:
            # Test
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        self.epsilon *= 0.99
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # Stop q_value update

    def getLegalActions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def pickAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)
        return self.computeActionFromQValues(state)

    def getAction(self, state):
        """
            Simply calls the pickAction method and then
            record the action taken for next next reward computation.
        """
        action = self.pickAction(state)
        self.record(state, action)
        return action

    def record(self, state, action):
        """ Record (state, action) pair of last run, for updating q value
        :param state: GameState
        :param action: Directions
        """
        self.lastState = state
        self.lastAction = action

    def turnoff_training(self):
        self.epsilon = 0
        self.alpha = 0
        self.shareQ = False


class ApproxQLearningGhost(AbstractQLearningGhost):
    def __init__(self, feature_extractor="GhostFeatureExtractor", **kwargs):
        self.feature_extractor = util.lookup(feature_extractor, globals())()
        self.weights = util.Counter()
        super(ApproxQLearningGhost, self).__init__(**kwargs)

    def export_data(self):
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "partialObs": self.partialObs,
            "weights": shared_weights if self.shareQ else self.weights
        }
        return data

    def getWeights(self):
        """
        :return: weights for computing approximate q value
        """
        if self.shareQ:
            return shared_weights
        else:
            return self.weights

    def getQValue(self, state, action):
        """
        :param state: GameState
        :param action: Directions
        :return: approximate Q value
        """
        features = self.feature_extractor.get_features(state, action, self.index)
        return self.getWeights() * features

    def updateQValue(self, state, action, next_state, reward):
        next_state_value = self.getValue(next_state)
        current_q_value = self.getQValue(state, action)
        difference = reward + self.gamma * next_state_value - current_q_value
        features = self.feature_extractor.get_features(state, action, self.index)
        for key in features.keys():
            update = self.alpha * difference * features[key]
            if self.shareQ:
                shared_weights[key] += update
            else:
                self.weights[key] += update


class ExactQLearningGhost(AbstractQLearningGhost):
    def export_data(self):
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "partialObs": self.partialObs,
            "q_values": shared_q if self.shareQ else self.q_values
        }
        return data

    def getQValue(self, state, action):
        if self.shareQ:
            return shared_q[(state, action)]
        else:
            return self.q_values[(state, action)]

    def updateQValue(self, state, action, next_state, reward):
        next_state_value = self.getValue(next_state)
        current_q_value = self.getQValue(state, action)
        update = self.alpha * (
                reward + self.gamma * next_state_value - current_q_value)
        if self.shareQ:
            shared_q[(state, action)] += update
        else:
            self.q_values[(state, action)] += update
