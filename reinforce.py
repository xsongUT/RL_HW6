from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

        self.network = nn.Sequential(

            nn.Linear(state_dims, 32),

            nn.ReLU(),

            nn.Linear(32, 32),

            nn.ReLU(),

            nn.Linear(32, num_actions),

            nn.Softmax(dim=-1)

        )

        self.optimizer = optim.Adam(self.network.parameters(), 

                                  lr=alpha, 

                                  betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        # TODO: implement this method

            with torch.no_grad():

                s = torch.FloatTensor(s)

                probs = self.network(s)

                action = torch.multinomial(probs, 1).item()

            return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method

        s = torch.FloatTensor(s)

        probs = self.network(s)

        log_prob = torch.log(probs[a])

        loss = -gamma_t * delta * log_prob

        

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        super().__init__(0)

        self.network = nn.Sequential(

            nn.Linear(state_dims, 32),

            nn.ReLU(),

            nn.Linear(32, 32),

            nn.ReLU(),

            nn.Linear(32, 1)

        )

        self.optimizer = optim.Adam(self.network.parameters(), 

                                  lr=alpha, 

                                  betas=(0.9, 0.999))



    def __call__(self,s) -> float:
        # TODO: implement this method

        with torch.no_grad():

            s = torch.FloatTensor(s)

            return self.network(s).item()

    def update(self,s,G):
        # TODO: implement this method

        s = torch.FloatTensor(s)

        value = self.network(s)

        loss = 0.5 * (G - value)**2

        

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method

    G_0 = []

    

    for episode in range(num_episodes):

        states, actions, rewards = [], [], []

        state = env.reset()

        done = False

        

        # Collect trajectory

        while not done:

            action = pi(state)

            next_state, reward, done, _ = env.step(action)

            

            states.append(state)

            actions.append(action)

            rewards.append(reward)

            

            state = next_state

            

        # Calculate returns and update

        G = 0

        returns = []

        for r in reversed(rewards):

            G = r + gamma * G

            returns.insert(0, G)

        G_0.append(returns[0])

        

        # Update policy and value function

        for t in range(len(states)):

            state = states[t]

            action = actions[t]

            G_t = returns[t]

            

            # Calculate baseline

            baseline = V(state)

            # Calculate advantage

            delta = G_t - baseline

            # Update policy

            pi.update(state, action, gamma**t, delta)

            # Update baseline

            V.update(state, G_t)

            

    return G_0
