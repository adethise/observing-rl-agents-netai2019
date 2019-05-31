import numpy as np
import sys
import argparse
import pickle

from lime.lime_tabular import LimeTabularExplainer



MODEL = 'model/pretrain_linear_reward.ckpt'
HEADERFILE = 'headers.csv'


class Explanation:
    def __init__(self, datafile, headerfile = HEADERFILE, model = MODEL):
        """
        Build an explanation for one run.
        """
        self.model = MODEL
        self.trace = datafile

        # Format the CSV data
        self._data = Explanation.format(datafile, headerfile)

        timesteps = self._data['timesteps']
        features_values = self._data['features_values']
        features_names = self._data['features_names']
        classes_names = self._data['classes_names']

        self.features = features_names
        self._num_base_features = len(features_names)
        self.classes = classes_names
        self.length = timesteps
        self._values = features_values
        self.rewards = self._data['rewards']
        self.rebufs = self._data['rebufs']

        # Load the neural network predictor and build the explainer
        from predictor import Predictor
        predictor = Predictor(MODEL)
        #explainer = LimeTabularExplainer(features_values,
        explainer = LimeTabularExplainer(pickle.load(open('explainer_train_data.pckl', 'rb')),
                                         feature_names = features_names,
                                         class_names = classes_names)

        self._explanations = list()

        # Initialize the weights
        # Weights are stored in an array with one line per timestep, on column per feature
        self._weights = np.zeros((timesteps, len(self.features)))
        self._weights_by_class = np.zeros((timesteps, len(self.features), len(self.classes)))
        self._class_probs = np.zeros((timesteps, len(self.classes)))

        # Explain each prediction (one per timestep)
        for time in range(timesteps):
            state = self._values[time, :]

            print(f'Explaining instance {time}...')
            explanation = explainer.explain_instance(state,
                                                     predictor.probs,
                                                     top_labels = len(self.classes),
                                                     num_features = len(self.features))
            # Store explanation
            weights = explanation.as_map()
            self._class_probs[time, :] = explanation.predict_proba

            # Compute the weight of each feature
            for class_ in weights:
                for feature, weight in weights[class_]:
                    # For each feature, compute the sum across all classes
                    self._weights[time, feature] += abs(weight)
                    # And save the weight by class
                    self._weights_by_class[time, feature, class_] = weight

            # Normalize
            self._weights[time, :] /= sum(self._weights[time, :])

    @property
    def base_features(self):
        return self.features[:self._num_base_features]


    @property
    def compound_features(self):
        return self.features[self._num_base_features:]


    def merge_features(self, sources, destination, method = 'avg'):
        agg = {'sum': sum, 'max': max, 'avg': lambda x: sum(x) / len(x)}[method]

        # use sum for weights and agg for values
        newrow_w = np.reshape(sum([self.weights_over_time(feature) for feature in sources]), (self.length, 1))
        self._weights = np.hstack([self._weights, newrow_w])

        newrow_wbc = np.reshape(sum([self.weights_by_class_over_time(feature) for feature in sources]), (self.length, 1, -1))
        self._weights_by_class = np.hstack([self._weights_by_class, newrow_wbc])

        newrow_v = np.reshape(agg([self.values_over_time(feature) for feature in sources]), (self.length, 1))
        self._values = np.hstack([self._values, newrow_v])

        self.features = np.append(self.features, destination)

    def feature_to_index(self, feature):
        return np.where(self.features == feature)[0][0]


    def weights_over_time(self, feature):
        '''
        Return the weight of a feature over time in this experiment.
        The weight is computed as the absolute sum of weights for each class.

        :param feature: the name of the feature
        :return: a 1D array of floats representing the weights
        '''
        return self._weights[:, self.feature_to_index(feature)]


    def weights_by_class_over_time(self, feature):
        '''
        Return the weight of a feature over time in this experiment for each class.

        :param feature: the name of the feature
        :return: a 2D array of floats representing the weights for each class (see self.classes)
        '''
        return self._weights_by_class[:, self.feature_to_index(feature), :]


    def weights_at_time(self, timestep):
        '''
        Return the weight of each feature at a given time in this experiment.

        :param timestep: the time at which to collect weights
        :return: a dict of {feature_name: float} representing the weights
        '''
        return {feature: weight for feature, weight in zip(self.features, self._weights[timestep, :])}


    def values_over_time(self, feature):
        '''
        Return the values of a feature over time in this experiment.

        :param feature: the name of the feature
        :return: a 1D array of floats representing the values
        '''
        return self._values[:, self.feature_to_index(feature)]


    def values_at_time(self, timestep):
        '''
        Return the value of each feature at a given time in this experiment.

        :param timestep: the time at which to collect weights
        :return: a dict of {feature_name: float} representing the values
        '''
        return {feature: value for feature, value in zip(self.features, self._values[timestep, :])}


    def probs(self, timestep):
        '''
        Return the probability of each class at a given time

        :param feature: the name of the feature
        :return: a 1D array of floats representing the probabilities (see self.classes)
        '''
        return self._class_probs[timestep, :]


    def probs_over_time(self):
        '''
        Return the probability of each class at each timestep.

        :param feature: the name of the feature
        :return: a 2D array of representing the classes (dim 1) and time (dim 2)
        '''
        return np.transpose(self._class_probs)


    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))


    @staticmethod
    def format(datafile, headfile):
        data = np.loadtxt(open(datafile, 'r'), delimiter = ',')
        headers = np.array(open(headfile, 'r').readline().strip().split(','))

        timesteps, _ = data.shape

        classes = data[:, 0] # selected class
        rewards = data[:, 1] # reward at each time step
        rebufs = data[:, 2] # rebuffering time at each time step
        classes_prob = data[:, 3:9]
        classes_names = [f'{br}kbps' for br in [300, 750, 1200, 1850, 2850, 4300]]

        epochs = data[:, 9] # epoch for which those features have been selected

        def is_valid(feature):
            if feature == 'last_quality_7':
                return True
            if feature == 'buffer_7':
                return True
            if feature.startswith('throughput'):
                return True
            if feature.startswith('latency'):
                return True
            if feature.startswith('next_size') and int(feature[-1]) < 6:
                return True
            if feature == 'remaining_chunks_7':
                return True
            return False

        feature_selector = list(map(is_valid, headers))
        features = data[:, feature_selector]
        features_names = headers[feature_selector]


        return {
                'timesteps': timesteps,
                'classes': classes,
                'rewards': rewards,
                'rebufs': rebufs,
                'classes_prob': classes_prob,
                'classes_names': classes_names,
                'epochs': epochs,
                'features_values': features,
                'features_names': features_names
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create explanations of a state trace')
    parser.add_argument('trace')
    parser.add_argument('output_file', help = 'save the model to a file for fast reloading', nargs = '?')

    args = parser.parse_args()

    datafile = args.trace
    explanation = Explanation(datafile)

    if args.output_file:
        explanation.save(args.output_file)
