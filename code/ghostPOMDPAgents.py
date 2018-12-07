from ghostAgents import GhostAgent
from ghostRLAgents import QLearningGhost
import util, time, random


class POMDPGhost(GhostAgent):
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
        self.record = []
        self.transition = util.Counter()

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
        self.transition[e1, action, e2] = self.transition[e1, action, e2] + 1

    def getAction(self, state):
        """
            record the action taken for next next reward computation.
        """
        # Initialize belief distribution if belief distribution not exists
        if self.belief == None:
            e = state.pacman
            self.belief = {}
            self.belief[e] = 0.6
            self.belief[(e[0]-1,e[1])] = 0.1
            self.belief[(e[0]+1,e[1])] = 0.1
            self.belief[(e[0],e[1]-1)] = 0.1
            self.belief[(e[0],e[1]+1)] = 0.1

        actions = self.getLegalActions(state)
        state_to_evidence = util.Counter()
        state_to_action = {}
        for k in self.belief.keys():
            v = util.Counter()
            for a in actions:
                result = util.Counter()
                for t in self.transition.keys():
                    if t[0] == k and t[1] == a:
                        result[t[2]] = self.transition[t]
                        state_to_evidence[t[2]] = self.transition[t] + state_to_evidence[t[2]]
                if len(result.keys()) == 0:
                    v[a] = 0
                else:
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

        value = float('-Inf')
        for a in v.keys():
            if v[a] > value:
                action = a



        if self.belief != None:
            e = state.pacman
            if e != None:
                nextState = state_to_evidence.normalize()
                evidence = util.Counter()
                transfer = {}
                transfer[(0,0)] = 0.6
                transfer[(1,0)] = 0.1
                transfer[(-1,0)] = 0.1
                transfer[(0,1)] = 0.1
                transfer[(0,-1)] = 0.1
                if nextState != None:
                    for s in nextState.keys():
                        x = e[0] - s[0]
                        y = e[1] - s[1]
                        if (x,y) in transfer.keys():
                            evidence[s] = transfer[(x,y)] * nextState[s] + evidence[s]
                evidence.normalize()
                self.belief = evidence
                if len(evidence.keys()) == 0:
                    self.belief = {}
                    self.belief[e] = 0.6
                    self.belief[(e[0]-1,e[1])] = 0.1
                    self.belief[(e[0]+1,e[1])] = 0.1
                    self.belief[(e[0],e[1]-1)] = 0.1
                    self.belief[(e[0],e[1]+1)] = 0.1

        # Update records for evidence and action
        if len(self.record) > 0:
            self.update(self.record[-1][0], self.record[-1][1], e)
        self.record.append((e,a))
        return action

