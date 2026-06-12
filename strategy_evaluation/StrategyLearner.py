""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  
Student Name: Lina Brihoum
GT User ID: lbrihoum3
GT ID: 903261368  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""

import datetime as dt
import pandas as pd
import util as ut
import QLearner as ql
import marketsimcode as ms
import indicators as ind


class StrategyLearner(object):
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.verbose = verbose  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.impact = impact  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.commission = commission
        self.q_learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.8, radr=0.9)

    # this method should create a QLearner, and train it for trading  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def add_evidence(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=10000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ):
        self.sv = sv

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]

        window = 10

        bollinger_band = ind.bollinger_band(symbol=syms, sd=sd, ed=ed, window=window)
        sma = ind.simple_moving_avg(symbol=syms, sd=sd, ed=ed, window=window)
        momentum = ind.momentum(symbol=syms, sd=sd, ed=ed, window=window)

        bollinger_band_samples = pd.qcut(bollinger_band[(window - 1):], window, labels=False)
        sma_samples = pd.qcut(sma[(window - 1):], window, labels=False)
        momentum_samples = pd.qcut(momentum[(window - 1):], window, labels=False)

        data = pd.DataFrame(100 * bollinger_band_samples + 10 * sma_samples + momentum_samples, index=bollinger_band_samples.index, columns=syms)

        self.q_learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.8, radr=0.9)

        df_trades = prices.copy()
        df_trades[:] = 0

        converged = False
        count = 0
        portval_prev = self.sv

        while not converged:
            df_trades[:] = 0
            q = self.q_learner.querysetstate(data.ix[0].values[0])

            if q == 0:
                holding = -1000
                df_trades.ix[window-1] = holding
            elif q == 1:
                holding = 0
                df_trades.ix[window-1] = holding
            elif q == 2:
                holding = 1000
                df_trades.ix[window-1] = holding

            for i in range(1, data.shape[0]):
                if q == 0:
                    impact_mult = -1
                elif q == 1:
                    impact_mult = 0
                elif q == 2:
                    impact_mult = 1

                q = self.q_learner.query(data.ix[i].values[0], (holding * ((prices.ix[data.index[i]].values[0]/prices.ix[data.index[i-1]].values[0])-1 - self.impact*impact_mult)))

                if q == 0:
                    if holding == 0:
                        df_trades.loc[data.index[i]] = -1000.0
                    elif holding == 1000:
                        df_trades.loc[data.index[i]] = -2000.0
                elif q == 2:
                    if holding == 0:
                        df_trades.loc[data.index[i]] = 1000.0
                    elif holding == -1000:
                        df_trades.loc[data.index[i]] = 2000.0

                holding = int(holding + df_trades.loc[data.index[i]].values[0])

            portval = ms.compute_portvals(df_trades, start_val=self.sv, commission=0, impact=0)

            if abs(portval[-1] - portval_prev) < 100 and count > 30:
                converged = True
            count += 1

            portval_prev = portval[-1].copy()

    def testPolicy(
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=10000,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    ):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        window = 10

        bollinger_band = ind.bollinger_band(symbol=syms, sd=sd, ed=ed, window=window)
        sma_indicator = ind.simple_moving_avg(symbol=syms, sd=sd, ed=ed, window=window)
        momentum_indicator = ind.momentum(symbol=syms, sd=sd, ed=ed, window=window)

        bollinger_band_samples = pd.qcut(bollinger_band[(window - 1):], window, labels=False)
        sma_samples = pd.qcut(sma_indicator[(window - 1):], window, labels=False)
        momentum_samples = pd.qcut(momentum_indicator[(window - 1):], window, labels=False)

        data = pd.DataFrame(100 * bollinger_band_samples + 10 * sma_samples + momentum_samples,
                              index=bollinger_band_samples.index, columns=syms)

        df_trades = prices.copy()
        df_trades[:] = 0
        cash = 0
        for i in range(0, data.shape[0]):
            q = self.q_learner.querysetstate(data.ix[i].values[0])

            if q == 0:
                if cash == 0:
                    df_trades.loc[data.index[i]] = -1000.0
                elif cash == 1000:
                    df_trades.loc[data.index[i]] = -2000.0
            elif q == 2:
                if cash == 0:
                    df_trades.loc[data.index[i]] = 1000.0
                elif cash == -1000:
                    df_trades.loc[data.index[i]] = 2000.0

            cash = int(cash + df_trades.loc[data.index[i]].values[0])

        return df_trades

    def author(self):
        return 'lbrihoum3'