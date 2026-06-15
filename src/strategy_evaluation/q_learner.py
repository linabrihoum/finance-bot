from __future__ import annotations

import numpy as np
import random as rand


class QLearner:
    """
    Tabular Q-learning agent with optional Dyna-Q experience replay.

    The agent maps discrete (state, action) pairs to expected cumulative
    rewards via a Q-table updated with the Bellman equation on each step.
    Exploration is controlled by a decaying random action rate (rar).

    Dyna-Q mode adds `dyna` additional hallucinated Q-updates per real step
    by sampling from a learned transition model. This lets the agent re-use
    past experience and converges significantly faster than vanilla Q-learning
    when environment interactions are expensive.

    Actions are indexed 0 to num_actions - 1.
    States are discrete integers in [0, num_states - 1].
    """

    def __init__(
        self,
        num_states: int = 100,
        num_actions: int = 4,
        alpha: float = 0.2,
        gamma: float = 0.9,
        rar: float = 0.5,
        radr: float = 0.99,
        dyna: int = 0,
        verbose: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        num_states : int
            Total number of discrete states in the environment

        num_actions : int
            Total number of discrete actions available to the agent

        alpha : float
            Learning rate for Q-table updates. Higher values weight recent experience more heavily

        gamma : float
            Discount factor for future rewards. Values close to 1 make the agent plan further into the future

        rar : float
            Initial random action rate — probability of choosing a random
            action instead of the greedy one (epsilon in ε-greedy policy)

        radr : float
            Multiplicative decay applied to rar after each call to query().
            Drives exploration toward exploitation over time.

        dyna : int
            Number of hallucinated Dyna-Q updates per real step.
            Set to 0 to disable Dyna-Q.

        verbose : bool
            Print debug output when True

        random_state : int
            Seed for both the Python random module and NumPy, ensuring reproducible episode trajectories

        """
        rand.seed(random_state)
        np.random.seed(random_state)

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # last observed state and action — carried between query() calls
        self.s: int = 0
        self.a: int = 0

        self.Q = np.zeros((num_states, num_actions))
        self.R = np.zeros((num_states, num_actions))

        # T[s, a, s'] counts observed transitions; uniform prior of 1 avoids
        # division-by-zero when normalizing into transition probabilities
        self.T = np.ones((num_states, num_actions, num_states))

    def querysetstate(self, s: int) -> int:
        """
        Set the current state and return an action without updating Q-table

        Used at the start of an episode or when running the final greedy policy without affecting the learned values

        Parameters
        ----------
        s : int
            The new state index

        Returns
        -------
        int
            The selected action — random with probability rar, greedy otherwise

        """
        action = (
            rand.randint(0, self.num_actions - 1)
            if rand.random() <= self.rar
            else int(np.argmax(self.Q[s, :]))
        )
        self.s = s
        self.a = action
        return action

    def query(self, s_prime: int, r: float) -> int:
        """
        Update the Q-table for the last transition and return the next action

        Applies the Bellman equation to update Q[s, a] using the received
        reward and the estimated value of the successor state. Optionally
        runs Dyna-Q hallucinated updates before choosing the next action.

        Parameters
        ----------
        s_prime : int
            The state reached after the agent's last action

        r : float
            The immediate reward received for that transition

        Returns
        -------
        int
            The next action to take from state s_prime

        """

        # Bellman update: Q[s,a] ← (1-α)·Q[s,a] + α·(r + γ·max_a' Q[s',a'])
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
            r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])]
        )

        action = (
            rand.randint(0, self.num_actions - 1)
            if rand.uniform(0.0, 1.0) <= self.rar
            else int(np.argmax(self.Q[s_prime, :]))
        )

        if self.dyna > 0:
            # Normalize transition counts to empirical probabilities before
            # recording the current transition so hallucinated updates use
            # the model state from before this step.
            transition_probs = self.T / np.sum(self.T)
            self.T[self.s, self.a, s_prime] += 1
            # Running-average update for the expected reward model
            self.R[self.s, self.a] += self.alpha * (r - self.R[self.s, self.a])

            for _ in range(self.dyna):
                rs = rand.randint(0, self.num_states - 1)
                ra = rand.randint(0, self.num_actions - 1)
                
                # Sample next state from the learned transition distribution
                next_s = int(np.argmax(transition_probs[rs, ra, :]))

                # Hallucinated Bellman update using the reward and transition models
                self.Q[rs, ra] = (1 - self.alpha) * self.Q[rs, ra] + self.alpha * (
                    self.R[rs, ra] + self.gamma * self.Q[next_s, np.argmax(self.Q[next_s, :])]
                )

        self.rar *= self.radr  # decay exploration rate toward greedy behavior
        self.s = s_prime
        self.a = action

        return action
