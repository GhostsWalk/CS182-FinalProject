from ghostAgents import GhostAgent
import util, time, random


class QLearningGhost(GhostAgent):
    def __init__(self, index, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1, partialObs=False):
        """
        :param actionFn: Function which takes a state and returns the list of legal actions
        :param numTraining: number of training episodes, i.e. no learning after these many episodes
        :param epsilon: exploration rate
        :param alpha: learning rate
        :param gamma: discount factor
        """
        self.index = index
        if actionFn is None:
            actionFn = lambda state: state.getLegalActions(agentIndex=index)
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.partialObs = partialObs

        self.qValues = util.Counter()

    def startEpisode(self):
        """
            Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
            Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def getLegalActions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        if not self.lastState is None:
            reward = -(state.getScore() - self.lastState.getScore()) # Ghost score is the reverse of Pacman's
            self.observeTransition(self.lastState, self.lastAction, obs, reward)
        return obs

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print 'Beginning %d episodes of Training' % self.numTraining

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = -(state.getScore() - self.lastState.getScore()) # Ghost score is the reverse of Pacman's
        # print("reward {}".format(deltaReward))
        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        self.observeTransition(self.lastState, self.lastAction, obs, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += -state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print 'Reinforcement Learning Status for agent {}: '.format(self.index)
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print '\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining)
                print '\tAverage Rewards over all training: %.2f' % (
                        trainAvg)
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
                print '\tAverage Rewards over testing: %.2f' % testAvg
            print '\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg)
            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))

        # print("q values: {}".format(self.qValues.values()))
        # print("qvalues: {}".format(self.qValues.keys()))

    ###############################
    # Q-learning Specific Updates #
    ###############################

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_value = self.computeValueFromQValues(nextState)
        value = self.getQValue(state, action)
        self.qValues[(state, action)] += self.alpha * (reward +
                                                        self.discount *
                                                        next_value - value)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        best_value = max([self.getQValue(state, action) for action in
                          actions])
        return best_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = list(self.getLegalActions(state))
        if len(actions) == 0:
            return None

        random.shuffle(actions)
        best_action = None
        best_value = None
        for action in actions:
            value = self.getQValue(state, action)
            if best_value is None or value > best_value:
                best_action = action
                best_value = value
        return best_action

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
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def getAction(self, state):
        """
            Simply calls the pickAction method and then
            record the action taken for next next reward computation.
        """
        action = self.pickAction(state)
        self.recordAction(state, action)
        return action

    def recordAction(self, state, action):
        self.lastState = state
        self.lastAction = action
