"""
Project 8: Strategy Evaluation
Student Name: Lina Brihoum
GT User ID: lbrihoum3
GT ID: 903261368
ML4T CS7646
testproject.py
"""

import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2
import StrategyLearner as sl
import datetime as dt


def author():
    return 'lbrihoum3'


if __name__ == '__main__':
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    sv = 100000
    impact = 0
    commission = 0

    exp1.experiment1(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), symbol = "JPM", sv = 100000, impact = 0, commission = 0)

    learner = sl.StrategyLearner(verbose=False, impact=impact)
    learner.add_evidence(symbol = "JPM", sd=sd, ed=ed, sv=sv)
    s_trades = learner.testPolicy(symbol = "JPM", sd=sd, ed=ed)

    exp2.experiment2(s_trades, sv, 0)
    exp2.experiment2(s_trades, sv, 0)

    manualLearner = ms.ManualStrategy()
    manualLearner.MainManualStrategy()

    learner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)  # constructor
    learner.add_evidence(symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                         sv=100000)  # training phase
    learner.testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                   sv=100000)