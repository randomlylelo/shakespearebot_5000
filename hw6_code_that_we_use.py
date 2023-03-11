import re


def parse_observations(lines):
    # Convert text to dataset.
    # lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []

        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1

            # Add the encoded word.
            obs_elem.append(obs_map[word])

        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''
        # self.D is the encoding size for x tokens
        # self.L is the encoding size for y tokens

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, log_probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.

        # rows represent the sequence length
        # cols are a values. that is, they represent the last token in the sequence
        log_probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # rows represent the sequence length
        # cols are a values. that is, they represent the last token in the sequence
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        ###

        # (a,b) represents P(y^j = a | y^{j-1} = b)
        # rows = last POS
        # cols = cur POS
        # transition_matrix[row][col] = P(cur POS | last POS)
        # self.A is transition matrix

        # also called "emission probabilities"
        # rows = POS
        # cols = word
        # observation_matrix[row][col] = P(word | POS)
        # self.O is observation matrix

        # observation_matrix = [] # (w,z) represents P(x^j = w | y^j = z)

        for hidden_state in range(self.L):
            # Most likely 1-length sequence to end in `hidden_state`
            # is just '{hidden_state}'
            seqs[1][hidden_state] = str(hidden_state)

            # self.A_start[hidden_state] gives the probability of going from
            # start state (i.e. we have nothing) to having `hidden_state`.
            # e.g. prob of going from '' to '2' is `self.A_start[2]`
            # log_probs[1][hidden_state] is probability of getting the sequence
            # '{hidden_state}' e.g. log_probs[1][3] is probability of getting
            # '3'. so, these are the same probability (we just need to log
            # one of them)
            zeroth_observation = x[0]
            log_probs[1][hidden_state] = np.log(
                self.A_start[hidden_state] * self.O[hidden_state][zeroth_observation])

        for seq_len in range(2, M + 1):
            for hidden_state in range(self.L):
                prev_seq_len = seq_len - 1

                best_prev_hidden_state = -1
                best_cur_log_prob = 0

                # Compute the argmax
                for prev_hidden_state in range(self.L):
                    prev_log_prob = log_probs[prev_seq_len][prev_hidden_state]

                    # Prob of transitioning FROM prev TO cur
                    transition_log_prob = np.log(
                        self.A[prev_hidden_state][hidden_state])

                    # Prob of emitting observiation `seq_len` state `hidden_state`
                    jth_token = x[seq_len - 1]
                    emission_log_prob = np.log(self.O[hidden_state][jth_token])

                    # We add since we're working with log probs
                    cur_log_prob = prev_log_prob + transition_log_prob + emission_log_prob

                    # If we found something better than our cur best answer,
                    # overwrite the cur best answer
                    if best_prev_hidden_state == -1 or cur_log_prob > best_cur_log_prob:
                        best_prev_hidden_state = prev_hidden_state
                        best_cur_log_prob = cur_log_prob

                if best_prev_hidden_state == -1:
                    raise Exception(
                        'did not find any hidden state to overwrite')

                best_prev_seq = seqs[prev_seq_len][best_prev_hidden_state]

                seqs[seq_len][hidden_state] = best_prev_seq + str(hidden_state)
                log_probs[seq_len][hidden_state] = best_cur_log_prob

        best_last_seq_idx = np.argmax(log_probs[M])
        max_seq = seqs[M][best_last_seq_idx]

        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###

        # Base cases
        for hidden_state in range(self.L):
            zeroth_observation = x[0]
            alphas[1][hidden_state] = self.A_start[hidden_state] * \
                self.O[hidden_state][zeroth_observation]

        # wrong
        for j in range(2, M + 1):
            for hidden_state in range(self.L):
                jth_token = x[j - 1]
                p_x = self.O[hidden_state][jth_token]

                total_sum = 0

                for other_hidden_state in range(self.L):
                    # other_hidden_state is a' in HMM_notes.pdf
                    alpha_term = alphas[j - 1][other_hidden_state]
                    p_y = self.A[other_hidden_state][hidden_state]

                    total_sum += alpha_term * p_y

                alphas[j][hidden_state] = p_x * total_sum

            if normalize:
                normalization_factor = 1 / sum(alphas[j])
                for hidden_state in range(self.L):
                    alphas[j][hidden_state] = alphas[j][hidden_state] * \
                        normalization_factor

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###

        # Base cases
        for hidden_state in range(self.L):
            betas[M][hidden_state] = 1

        for j in range(M-1, 0, -1):
            for hidden_state in range(self.L):
                jth_token = x[j]

                total_sum = 0

                for other_hidden_state in range(self.L):
                    # other_hidden_state is b' in HMM_notes.pdf

                    beta_term = betas[j + 1][other_hidden_state]

                    p_y = self.A[hidden_state][other_hidden_state]

                    p_x = self.O[other_hidden_state][jth_token]

                    total_sum += beta_term * p_y * p_x

                betas[j][hidden_state] = total_sum

            if normalize:
                normalization_factor = 1 / sum(betas[j])
                for hidden_state in range(self.L):
                    betas[j][hidden_state] = betas[j][hidden_state] * \
                        normalization_factor

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                # for each value in A
                numerator = 0
                denominator = 0

                for i in range(len(Y)):
                    M = len(Y[i])
                    for k in range(1, M):
                        # I think it should be M+1 but gives list out of range.
                        # Makes sense maybe from the formulas, if we shift
                        # everything 1.
                        if Y[i][k] == b and Y[i][k - 1] == a:
                            numerator += 1

                        if Y[i][k-1] == a:
                            denominator += 1

                # flipped
                self.A[a][b] = numerator/denominator

        # Calculate each element of O using the M-step formulas.

        for a in range(self.L):
            for w in range(self.D):
                # for each value in A
                numerator = 0
                denominator = 0

                for i in range(len(Y)):
                    M = len(Y[i])
                    for k in range(M):
                        if Y[i][k] == a and X[i][k] == w:
                            numerator += 1

                        if Y[i][k] == a:
                            denominator += 1

                # flipped
                self.O[a][w] = numerator/denominator

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        '''

        # For all equations, refer to HMM_notes.pdf

        N = len(X)

        """ Starter code provided by Jake Lee 2023-03-06 """
        for iteration in tqdm(range(1, N_iters + 1)):
            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_den = [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_num = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_den = [[0. for _ in range(self.D)] for _ in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)

                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # For P_curr, the i^th index is P(y^t = i, x).

                # j is the same as j in equation (12)
                for j in range(1, M + 1):
                    # Populate P_curr using equation (12)
                    P_curr = np.array([0. for _ in range(self.L)])

                    for curr in range(0, self.L):
                        P_curr[curr] = alphas[j][curr]*betas[j][curr]

                    if np.sum(P_curr) != 0:
                        P_curr /= np.sum(P_curr)

                    # Populate P_joint using equation (13)
                    P_joint = [[0. for _ in range(self.L)]
                               for _ in range(self.L)]

                    for curr in range(self.L):
                        for next in range(self.L):
                            if j < len(x):
                                P_joint[curr][next] = (
                                    alphas[j][curr] * self.A[curr][next] *
                                    self.O[next][x[j]] * betas[j + 1][next]
                                )

                    if np.sum(P_joint) != 0:
                        P_joint /= np.sum(P_joint)

                    # M: Update the A, O matrices.
                    # Update A matrix using equation (14)

                    # curr is a in equation (14)
                    for curr in range(self.L):
                        # next is b in equation (14)
                        for next in range(self.L):
                            # Because of the (k-1) term in equation (14),
                            # we must skip the last index
                            if j < len(x):
                                A_den[curr][next] += P_curr[curr]
                            A_num[curr][next] += P_joint[curr][next]

                    # Update O matrix using equation (15)
                    for hidden_state in range(self.L):
                        prev_obs = x[j - 1]

                        O_num[hidden_state][prev_obs] += P_curr[hidden_state]
                        O_den[hidden_state] += P_curr[hidden_state]

            self.O = np.array(O_num) / np.array(O_den)
            self.A = np.array(A_num) / np.array(A_den)

    def generate_emission(self, M, seed=None):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        # (Re-)Initialize random number generator
        rng = np.random.default_rng(seed=seed)

        emission = []
        states = []

        all_states = range(self.L)
        all_x_tokens = range(self.D)

        state = rng.choice(all_states)

        for i in range(M):
            # Add new state
            state = rng.choice(all_states, p=self.A[state])
            states.append(state)

            # Add new emission/observation
            observation = rng.choice(all_x_tokens, p=self.O[state])
            emission.append(observation)

        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]]
                    for j in range(self.L)])

        return prob


def unsupervised_HMM(X, n_states, N_iters, seed=None):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
        rng:        The random number generator for reproducible result.
                    Default to RandomState(1).
    '''
    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[rng.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[rng.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
