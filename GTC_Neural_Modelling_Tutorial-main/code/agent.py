import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon, eps_2, bonus):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon
        self.eps_2 = eps_2
        self.bonus = bonus

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        # complete the code
        experience_index = s * self.num_actions + a
        self.experience_buffer[experience_index, :] = [s, a, r, s1]
        return None

    def _update_qvals(self, s, a, r, s1, bonus = 0):

        '''
        Update the Q-value table
        Input arguments:
            s          -- initial state
            a          -- chosen action
            r          -- received reward
            s1         -- next state
        '''

        # complete the code
        e = np.max(self.Q[s1, :])

        if bonus == 1:
            bonus_term = self.epsilon * np.sqrt(self.action_count[s, a])
        elif bonus == 2:
            action_probs = self.action_count[s, :] / np.sum(self.action_count[s, :])
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            bonus_term = self.eps_2 * entropy        
        else:
            bonus_term = 0

        self.Q[s, a] += self.alpha * (r + bonus_term + self.gamma * e - self.Q[s, a])

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code
        self.action_count += 1
        self.action_count[s, a] = 0
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        # policy = Qxa + epsilon(nxa)^0.5
        policy = self.Q[s,:] + self.epsilon*np.sqrt(self.action_count[s,:])

        a = np.argmax(policy)
        max_indices = np.where(policy == a)[0]
        if len(max_indices) > 1:
            a = np.random.choice(max_indices)
        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for _ in range(num_planning_updates):
            # random experience from experience buffer
            rand_idx = np.random.randint(0, self.experience_buffer.shape[0])
            s, a, r, s1 = self.experience_buffer[rand_idx]
            # update q_vals
            self._update_qvals(s, a, r, s1, bonus =self.bonus)             

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus = 0)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        self.num_actions = 2

        return None
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    def _init_reward(self):

        '''
        Initialise rewards and boundaries
        '''

        self.rewards = np.random.uniform(0.25,0.75,4)
        self.bound = [0.25,0.75] # From paper


        return None
    
    def _init_q_values(self): 
        self.q_td = np.zeros((3,2))
        self.q_mb = np.zeros((3,2))
        self.q_net = np.zeros((3,2))
        return None
    
    def get_reward(self, s, a):
        r_pos = [[0, 1], [2, 3]]
        index = r_pos[s-1][a]
        p = self.rewards[index]

        r = np.random.choice([0,1], p = (1-p, p))
        return r
    
    def update_rewards(self):
        '''
        changes rewards by a gaussian noise
        and keeps it in boundaries
        '''
        self.rewards += np.random.normal(loc=0, scale=0.025, size=4)
        for i, reward in enumerate(self.rewards):
            if (reward < self.bound[0]):
                difference = self.bound[0] - reward
                self.rewards[i] += 2*difference

            elif (reward>self.bound[1]):
                difference = self.bound[1] - reward
                self.rewards[i] += 2*difference
    
    def _update_q_td(self, s, a, s_1, a_1, r):
        if s == 0: 
            delta = r + self.q_td[s_1, a_1] - self.q_td[s, a]
            self.q_td[s, a] += self.alpha1 * delta
        else:
            delta = r - self.q_td[s, a]
            self.q_td[s, a] += self.alpha2 * delta
            self.q_td[s-1, a] += self.alpha1 * delta * self.lam
  
        return None
    
    def _update_q_net(self, s, a):
        if s == 0: 
            self.q_net[s, a] = self.w * self.q_mb[s, a] + (1- self.w) * self.q_td[s, a]
        else: 
            self.q_net[s, a] = self.q_td[s, a]
        return None
    
    def _update_q_mb(self, s, a):
        probs = [[0.7, 0.3], [0.3, 0.7]]
        if s == 0:
            self.q_mb[s, a] = probs[0][a] * np.max(self.q_td[s+1, :]) + probs[1][a]*np.max(self.q_td[s + 2, :])
        else: 
            self.q_mb[s, a] = self.q_td[s, a]
        return None
  
    
    def _policy(self, s, last_a): 
        num = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            if s == 0:
                beta = self.beta1
                rep = last_a == a
            else: 
                beta = self.beta2
                rep = 0
            
            num[a] = np.exp(beta*(self.q_net[s, a] + self.p * rep))

        policy = num / np.sum(num)
        a = np.random.choice([0,1], p = policy)
        return a 
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]

        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])
     
    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
        self._init_q_values()
        self._init_history()
        self._init_reward()


        last_a= 8
        for i in range(num_trials): 
            states = [0, 1, 2]
            s = 0
            a_0 = self._policy(s, last_a)
            if a_0 == 0: 
                s_1 = np.random.choice([1, 2], p = [0.7, 0.3])
            else:
                s_1 = np.random.choice([1, 2], p = [0.3, 0.7])
            r = 0
            last_a = a_0
            a_1 = self._policy(s_1, last_a)

            # reward
            rew = self.get_reward(s, a_1)
            
            # Updates
            self._update_q_td(s, a_0, s_1, a_1, r) 
            self._update_q_td(s_1, a_1, '', '', rew)
            

            self._update_q_mb(s, a_0)
            self._update_q_mb(s_1, a_1)

            self._update_q_net(s, a_0)
            self._update_q_net(s_1, a_1)

            self._update_history(a_0, s_1, rew)
            self.update_rewards()
        
        return None