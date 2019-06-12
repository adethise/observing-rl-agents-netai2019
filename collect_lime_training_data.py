import os
import numpy as np
import pickle

import explainer

all_traces = os.listdir('exec_traces/normal_agent/')
values = [explainer.Explanation.format('exec_traces/normal_agent/' + trace, 'headers.csv')['features_values'] for trace in all_traces]

training_data = np.vstack(values)

pickle.dump(training_data, open('explainer_train_data.pckl', 'wb'))
