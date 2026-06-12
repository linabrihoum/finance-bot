"""
Project 8: Strategy Evaluation
Student Name: Lina Brihoum
GT User ID: lbrihoum3
GT ID: 903261368
ML4T CS7646
QLearner.py
"""

import numpy as np
import random as rand


class QLearner(object):
    def author(self):
        return 'lbrihoum3'

    def __init__(self, num_states=100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):

        self.verbose = verbose  		   	  			    		  		  		    	 		 		   		 		  
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.s = 0  		   	  			    		  		  		    	 		 		   		 		  
        self.a = 0

        self.Q = np.zeros((num_states, num_actions))
        self.R = np.zeros((num_states, num_actions))
        self.T = np.ones((num_states, num_actions, num_states))

    def querysetstate(self, s):
        # For initial state and for final policy. Will choose action randomly or maxarg
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """

        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])

        self.a = action
        self.s = s

        return action

    def query(self,s_prime,r):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
                    r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])  # update Q table

        if rand.uniform(0.0, 1.0) <= self.rar:  # choose random direction if random rate is higher than random num
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime, :])

        if self.dyna > 0:
            temp = (self.T / np.sum(self.T))
            self.T[self.s, self.a, s_prime] += 1
            self.R[self.s, self.a] += self.alpha * (r - self.R[self.s, self.a])
            for i in range(self.dyna):
                rand_1 = rand.randint(0, self.num_states - 1)
                rand_2 = rand.randint(0, self.num_actions - 1)
                # Update Board
                self.Q[rand_1, rand_2] = (1 - self.alpha) * self.Q[rand_1, rand_2] + self.alpha * (
                        self.R[rand_1, rand_2] + self.gamma * self.Q[
                    (np.argmax(temp[rand_1, rand_2, :])), np.argmax(
                        self.Q[(np.argmax(temp[rand_1, rand_2, :])), :])])

        self.rar *= self.radr  # update random rate for decay
        self.s = s_prime  # update state for next query
        self.a = action  # update action for next query

        return action
