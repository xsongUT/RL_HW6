import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here

        self.state_low = state_low

        self.state_high = state_high

        self.num_actions = num_actions

        self.num_tilings = num_tilings

        self.tile_width = tile_width

        

        #Row/Col number
        self.num_tiles_row = np.ceil((state_high[0] - state_low[0]) / tile_width[0]).astype(int) + 1 #location
        self.num_tiles_col = np.ceil((state_high[1] - state_low[1]) / tile_width[1]).astype(int) + 1 #velocity

        
        # Initialize offsets for each tiling
        self.offsets_row = np.array([(-i / num_tilings) * tile_width[0] for i in range(num_tilings)])
        self.offsets_col = np.array([(-i / num_tilings) * tile_width[1] for i in range(num_tilings)])

        
        # Total number of features (tiles) in a single tiling
        self.num_tiles = self.num_tiles_row *self.num_tiles_col 





    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.num_actions*self.num_tilings * self.num_tiles

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        """
        Generate a feature vector for a given state and action.
        If done is True, return a zero vector.
        """
        if done:
            return np.zeros(self.feature_vector_len())

        # Initialize feature vector
        feature_vector = np.zeros(self.feature_vector_len())
        
        
            
        for i in range(self.num_tilings):
            # Compute the ind for s in each tiling flat
            cell_ind_row =  ((s[0] - (self.state_low[0] + self.offsets_row[i] )) // self.tile_width[0]).astype(int)
            cell_ind_col =  ((s[1] - (self.state_low[1] + self.offsets_col[i] )) // self.tile_width[1]).astype(int)
            # Map indices to a unique position in the feature vector
            flat_index = np.ravel_multi_index(np.array([cell_ind_row,cell_ind_col]), (self.num_tiles_row,self.num_tiles_col))
            feature_index = a * self.num_tilings * self.num_tiles + i * self.num_tiles + flat_index
            feature_vector[feature_index] = 1

        return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
    for episode in range(num_episode):
        s = env.reset()
        a = epsilon_greedy_policy(s, False, w)

        # Initialize eligibility trace
        z = np.zeros_like(w)
        x = X(s, False, a)  # Feature vector for (s, a)
        Q_old = 0

        while True:
            # Take action a, observe reward r and next state s'
            s_next, r, done, _ = env.step(a)
            a_next = epsilon_greedy_policy(s_next, done, w) if not done else None

            # Compute feature vector for (s', a')
            x_next = X(s_next, done, a_next) if not done else np.zeros_like(x)

            # Compute TD error
            Q = np.dot(w, x)
            Q_next = np.dot(w, x_next) if not done else 0
            delta = r + gamma * Q_next - Q

            # Update eligibility trace
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x

            # Update weights
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_next

            if done:
                break

            # Move to the next state-action pair
            s, a, x = s_next, a_next, x_next

    return w