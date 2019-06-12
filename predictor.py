"""
Loads the agent and use it to make predictions on states.
"""

import tensorflow as tf
import a3c
import sys
import numpy as np


S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Predictor(metaclass = Singleton):

    def __init__(self, checkpoint):
        self.sess = tf.Session()

        self.actor = a3c.ActorNetwork(self.sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        self.saver.restore(self.sess, checkpoint)


    def probs(self, values):
        states = np.zeros((len(values), 6, 8))
        states[:, 0, -1] = values[:, 0]
        states[:, 1, -1] = values[:, 1]
        states[:, 2, :] = values[:, 2:10]
        states[:, 3, :] = values[:, 10:18]
        states[:, 4, :6] = values[:, 18:24]
        states[:, 5, -1] = values[:, 24]

        return np.array([np.reshape(self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN))), (6,)) for state in states])

    def _probs_no_reshape(self, states):
        return np.array([np.reshape(self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN))), (6,)) for state in states])

    def predict_max(self, state):
        action_prob = self.probs(state)
        return action_prob.argmax()

    def predict(self, state):
        action_prob = self.probs(state)
        action_cumsum = np.cumsum(action_prob)
        return (action_cumsum > np.random.random() ).argmax()


def predict_trace(checkpoint, trace_file, feature_filter = [], headfile = 'headers.csv'):
    data = np.loadtxt(open(trace_file, 'r'), delimiter = ',')
    headers = open(headfile, 'r').readline().strip().split(',')

    REPLACES = {
            'last_quality_7':      0.250097,
            'buffer_7':            1.573854,
            'throughput_7':        0.155318,
            'latency_7':           0.374323,
            'throughput_6':        0.155318,
            'latency_6':           0.374323,
            'throughput_0':        0.155318,
            'latency_5':           0.374323,
            'latency_3':           0.374323,
            'latency_0':           0.374323,
            'latency_2':           0.374323,
            'throughput_5':        0.155318,
            'latency_4':           0.374323,
            'throughput_3':        0.155318,
            'latency_1':           0.374323,
            'next_size_4':         1.437556,
            'next_size_5':         2.158495,
            'throughput_1':        0.155318,
            'throughput_2':        0.155318,
            'next_size_1':         0.380457,
            'remaining_chunks_7':  0.5,
            'next_size_3':         0.931950,
            'throughput_4':        0.155318,
            'next_size_2':         0.605353,
            'next_size_0':         0.153233,
            }

    for h in headers:
        if h not in feature_filter:
            index = headers.index(h)
            if h in REPLACES:
                data[:, index] = REPLACES[h]
            else: # default, use average in trace
                data[:, index] = sum(data[:, index]) / len(data[:, index])

    features = data[:, 10:]
    predictor = Predictor(checkpoint)
    probabilities = predictor._probs_no_reshape(features)

    tf.reset_default_graph()
    return probabilities


if __name__ == '__main__':
    predictor = Predictor(sys.argv[1])

    # Example state (debugging only)
    initial_state = np.array([[0.43023256, 0.27906977, 0.1744186 , 0.43023256, 0.27906977, 0.27906977, 0.43023256, 0.1744186 ],
                              [3.61143736, 3.54652224, 3.67964079, 3.55517278, 3.64901598, 3.71484625, 3.66065794, 3.89409025],
                              [0.12580341, 0.14414007, 0.15028695, 0.17998085, 0.15933404, 0.1722451 , 0.20006349, 0.23454729],
                              [0.70352783, 0.46491513, 0.26688145, 0.52446801, 0.3061568 , 0.33416973, 0.45418831, 0.16656769],
                              [0.147968  , 0.345516  , 0.566904  , 0.85845   , 1.3395    , 2.108491  , 0.144254  , 0.149991  ],
                              [0.3125    , 0.29166667, 0.27083333, 0.25      , 0.22916667, 0.20833333, 0.1875    , 0.16666667]])

