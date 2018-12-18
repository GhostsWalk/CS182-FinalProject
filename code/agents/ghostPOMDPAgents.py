from ghostAgents import GhostAgent
import util, time, random
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class BayesianRLGhost(GhostAgent):
    def __init__(self, index, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1, partialObs=True):
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
        self.belief = None
        self.record = []
        self.transition = util.Counter()

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

    def observeTransition(self, evidence, action, result, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, evidence, nextState, deltaReward)

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
        #deltaReward = -(state.getScore() - self.lastState.getScore()) # Ghost score is the reverse of Pacman's
        # print("reward {}".format(deltaReward))
        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        #self.observeTransition(self.lastState, self.lastAction, obs, deltaReward)
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


    ###############################
    # POMDP Specific Updates #
    ###############################

    def update(self, e1, action, e2):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.

          NOTE: You should never call this function,
          it will be called on your behalf
        """        
        self.transition[e1, action, e2] = self.transition[e1, action, e2] + 0.36
        for e in [(e2[0]-1,e2[1]), (e2[0]+1,e2[1]), (e2[0],e2[1]-1), (e2[0], e2[1]+1)]:
            self.transition[e1, action, e] = self.transition[e1, action, e] + 0.06
        for e in [(e1[0]-1,e1[1]), (e1[0]+1,e1[1]), (e1[0],e1[1]-1), (e1[0], e1[1]+1)]:
            self.transition[e, action, e2] = self.transition[e, action, e2] + 0.06
        for e_2 in [(e2[0]-1,e2[1]), (e2[0]+1,e2[1]), (e2[0],e2[1]-1), (e2[0], e2[1]+1)]:
            for e_1 in [(e1[0]-1,e1[1]), (e1[0]+1,e1[1]), (e1[0],e1[1]-1), (e1[0], e1[1]+1)]:
                self.transition[e_1,action,e_2] = self.transition[e_1,action,e_2] + 0.01

    def getAction(self, state):
        """
            record the action taken for next next reward computation.
        """

        # Get legal actions
        actions = self.getLegalActions(state)

        # Get values for specific state and action after 1 horizon

        # Initialize belief distribution if it does not exist
        if self.belief == None:
            e = state.pacman
            self.belief = util.Counter()
            self.belief[e] = 0.6
            self.belief[(e[0]-1,e[1])] = 0.1
            self.belief[(e[0]+1,e[1])] = 0.1
            self.belief[(e[0],e[1]-1)] = 0.1
            self.belief[(e[0],e[1]+1)] = 0.1
            self.belief.normalize()

        # Update belief if it exists
        else:
            e = state.pacman
            state_to_evidence = util.Counter()
            for k in self.belief.keys():
                a = self.record[-1][1]
                for t in self.transition.keys():
                    if t[0] == k and t[1] == a:
                        if e == t[2]:
                            state_to_evidence[t[2]] = self.transition[t] *0.6 + state_to_evidence[t[2]]
                        elif e == (t[2][0]-1, t[2][1]):
                            state_to_evidence[t[2]] = self.transition[t] *0.1 + state_to_evidence[t[2]]
                        elif e == (t[2][0]+1, t[2][1]):
                            state_to_evidence[t[2]] = self.transition[t] *0.1 + state_to_evidence[t[2]]
                        elif e == (t[2][0], t[2][1]-1):
                            state_to_evidence[t[2]] = self.transition[t] *0.1 + state_to_evidence[t[2]]
                        elif e == (t[2][0], t[2][1]+1):
                            state_to_evidence[t[2]] = self.transition[t] *0.1 + state_to_evidence[t[2]]
            if len(state_to_evidence.keys()) != 0:
                self.belief = state_to_evidence
                self.belief.normalize()
            else:
                self.belief = util.Counter()
                self.belief[e] = 0.6
                self.belief[(e[0]-1,e[1])] = 0.1
                self.belief[(e[0]+1,e[1])] = 0.1
                self.belief[(e[0],e[1]-1)] = 0.1
                self.belief[(e[0],e[1]+1)] = 0.1
                self.belief.normalize()


        # Update values of actions in horizon of 1 setp
        state_to_action = {}
        for k in self.belief.keys():
            v = util.Counter()
            for a in actions:
                result = util.Counter()
                for t in self.transition.keys():
                    if t[0] == k and t[1] == a:
                        result[t[2]] = self.transition[t]
                if len(result.keys()) > 0:
                    result.normalize()
                    for r in result.keys():
                        v[a] = v[a] + result[r]*0.6 *(-(abs(r[0])+abs(r[1]))) + result[r]*0.1 *(
                            -(abs(r[0]+1)+abs(r[1])))+ result[r]*0.1 *(-(abs(r[0]-1)+abs(r[1])))+ result[r]*0.1 *(
                            -(abs(r[0])+abs(r[1]+1))) + + result[r]*0.1 *(-(abs(r[0])+abs(r[1]-1)))
                state_to_action[(k,a)] = v[a]

        v = util.Counter()
        for a in actions:
            for k in self.belief.keys():
                v[a] = v[a] + self.belief[k] * state_to_action[(k,a)]

        action = None
        value = float('-Inf')
        for a in v.keys():
            if v[a] > value:
                action = a
                value = v[a]
            elif v[a] == value:
                p = random.random()
                if p < 0.5:
                    action = a

        # Update records for transitions and action
        e = state.pacman
        if len(self.record) > 0:
            self.update(self.record[-1][0], self.record[-1][1], e)
        self.record.append((e,a))
        return action


class MultiAgentGhost(GhostAgent):
    def __init__(self, index, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1, partialObs=True):
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
        self.belief = None
        self.record = []
        self.transition = util.Counter()

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

    def observeTransition(self, evidence, action, result, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, evidence, nextState, deltaReward)

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
        #deltaReward = -(state.getScore() - self.lastState.getScore()) # Ghost score is the reverse of Pacman's
        # print("reward {}".format(deltaReward))
        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        #self.observeTransition(self.lastState, self.lastAction, obs, deltaReward)
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


    ###############################
    # POMDP Specific Updates #
    ###############################

    def update(self, b1, action, b2):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        for k1 in b1.keys():
            for k2 in b2.keys():
                self.transition[k1, action, k2] = self.transition[k1, action, k2] + b1[k1] * b2[k2]       

    def getAction(self, state):
        """
            record the action taken for next next reward computation.
        """

        # Get legal actions
        actions = self.getLegalActions(state)

        # Get values for specific state and action after 1 horizon

        e = state.pacman
        g = state.ghosts[0]
        p = state.ghosts_to_pacman[0]
        e2 = (p[0]+g[0], p[1]+g[1])

        if e2 == e:
            self.belief = util.Counter()
            self.belief[e] = 0.6
            self.belief[(e[0]-1,e[1])] = 0.1
            self.belief[(e[0]+1,e[1])] = 0.1
            self.belief[(e[0],e[1]-1)] = 0.1
            self.belief[(e[0],e[1]+1)] = 0.1
            self.belief.normalize()
        else:
            self.belief = util.Counter()
            self.belief[e] = 0.5
            self.belief[e2] = 0.5
            self.belief.normalize()

        # Update values of actions in horizon of 1 setp
        state_to_action = {}
        for k in self.belief.keys():
            v = util.Counter()
            for a in actions:
                result = util.Counter()
                for t in self.transition.keys():
                    if t[0] == k and t[1] == a:
                        result[t[2]] = self.transition[t]
                if len(result.keys()) > 0:
                    result.normalize()
                    for r in result.keys():
                        v[a] = v[a] + result[r]*0.6 *(-(abs(r[0])+abs(r[1]))) + result[r]*0.1 *(
                            -(abs(r[0]+1)+abs(r[1])))+ result[r]*0.1 *(-(abs(r[0]-1)+abs(r[1])))+ result[r]*0.1 *(
                            -(abs(r[0])+abs(r[1]+1))) + + result[r]*0.1 *(-(abs(r[0])+abs(r[1]-1)))
                state_to_action[(k,a)] = v[a]

        v = util.Counter()
        for a in actions:
            for k in self.belief.keys():
                v[a] = v[a] + self.belief[k] * state_to_action[(k,a)]

        action = None
        value = float('-Inf')
        for a in v.keys():
            if v[a] > value:
                action = a
                value = v[a]
            elif v[a] == value:
                p = random.random()
                if p < 0.5:
                    action = a

        if action == None:
            random.shuffle(actions)
            action = actions[0]

        # Update records for transitions and action
        e = state.pacman
        if len(self.record) > 0:
            self.update(self.record[-1][0], self.record[-1][1], self.belief)
        self.record.append((self.belief,a))
        return action

class DeepRLGhost(GhostAgent):
    def __init__(self, index, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1, partialObs=True):
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
        self.belief = None
        self.record = []
        self.transition = util.Counter()

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

    def observeTransition(self, evidence, action, result, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, evidence, nextState, deltaReward)

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
        input_dim = 7
        H = 50
        model = models.Sequential()
        model.add(layers.Dense(H, input_dim = input_dim, kernel_initializer = 'normal', activation = 'linear'))
        model.add(layers.Dense(1, kernel_initializer = 'normal', activation = 'linear'))
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
        self.model = model

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        #deltaReward = -(state.getScore() - self.lastState.getScore()) # Ghost score is the reverse of Pacman's
        # print("reward {}".format(deltaReward))
        if self.partialObs:
            obs = state.obs(self.index)
        else:
            obs = state.deepCopy()
        #self.observeTransition(self.lastState, self.lastAction, obs, deltaReward)
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


    ###############################
    # POMDP Specific Updates #
    ###############################    
    def update(self, e1, action, e2):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.

          NOTE: You should never call this function,
          it will be called on your behalf
        """        
        self.transition[e1, action, e2] = self.transition[e1, action, e2] + 0.36
        for e in [(e2[0]-1,e2[1]), (e2[0]+1,e2[1]), (e2[0],e2[1]-1), (e2[0], e2[1]+1)]:
            self.transition[e1, action, e] = self.transition[e1, action, e] + 0.06
        for e in [(e1[0]-1,e1[1]), (e1[0]+1,e1[1]), (e1[0],e1[1]-1), (e1[0], e1[1]+1)]:
            self.transition[e, action, e2] = self.transition[e, action, e2] + 0.06
        for e_2 in [(e2[0]-1,e2[1]), (e2[0]+1,e2[1]), (e2[0],e2[1]-1), (e2[0], e2[1]+1)]:
            for e_1 in [(e1[0]-1,e1[1]), (e1[0]+1,e1[1]), (e1[0],e1[1]-1), (e1[0], e1[1]+1)]:
                self.transition[e_1,action,e_2] = self.transition[e_1,action,e_2] + 0.01


    def getAction(self, state):
        """
            record the action taken for next next reward computation.
        """

        # Get legal actions
        actions = self.getLegalActions(state)
        action_dict = {}
        action_dict['East'] = 0
        action_dict['West'] = 1
        action_dict['North'] = 2
        action_dict['South'] = 3
        action_dict['Stop'] = 4

        v = util.Counter()
        # Get values for specific state and action after 1 horizon
        if len(self.record) > 0:
            e = state.pacman
            g = state.ghosts[0]
            e_0 = self.record[-1][0]
            g_0 = self.record[-1][1]
            a_0 = action_dict[self.record[-1][2]]
            x = np.array([e[0],e[1],g_0[0],g_0[1],e_0[0],e_0[1],a_0])
            reward = self.record[-1][3] - state.getScore() 
            model = self.model
            model.fit(np.array([x]), np.array([reward]), verbose = 0)

            for a in actions:
                a_1 = action_dict[a]
                for t in self.transition.keys():
                    if (t[0] == e) & (t[1] == a_1):
                        e_1 = t[2]
                        x = np.array([e_1[0],e_1[1],g[0], g[1],e[0], e[1],a_1])
                        v[a_1] = v[a_1] + model.predict(np.array([x]))

        action = None
        value = float('-Inf')
        for a in v.keys():
            if v[a] > value:
                action = a
                value = v[a]
            elif v[a] == value:
                p = random.random()
                if p < 0.5:
                    action = a

        if action == None:
            random.shuffle(actions)
            action = actions[0]


        # Update records for transitions and action
        e = state.pacman
        r = state.getScore()
        g = state.ghosts[0]
        a = action
        self.record.append((e,g,a,r))
        # Update records for transitions and action
        e = state.pacman
        if len(self.record) > 0:
            self.update(self.record[-1][0], self.record[-1][2], e)
        return action
