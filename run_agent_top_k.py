import os
import sys

import pickle
import argparse

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf

import load_trace
import a3c
import env


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

FEATURES = [
        # ordered by feature importance
        # feature name          average value   index
        ('last_quality_7',      0.250097,       (0, -1)),
        ('buffer_7',            1.573854,       (1, -1)),
        ('throughput_7',        0.155318,       (2, -1)),
        ('throughput_6',        0.155318,       (2, -2)),
        ('latency_6',           0.374323,       (3, -2)),
        ('latency_7',           0.374323,       (3, -1)),
        ('throughput_0',        0.155318,       (2, -8)),
        ('latency_0',           0.374323,       (3, -8)),
        ('throughput_5',        0.155318,       (2, -3)),
        ('latency_5',           0.374323,       (3, -3)),
        ('latency_4',           0.374323,       (3, -4)),
        ('throughput_3',        0.155318,       (2, -5)),
        ('throughput_2',        0.155318,       (2, -6)),
        ('latency_3',           0.374323,       (3, -5)),
        ('latency_2',           0.374323,       (3, -6)),
        ('latency_1',           0.374323,       (3, -7)),
        ('throughput_1',        0.155318,       (2, -7)),
        ('next_size_5',         2.158495,       (4,  5)),
        ('next_size_4',         1.437556,       (4,  4)),
        ('next_size_1',         0.380457,       (4,  1)),
        ('next_size_3',         0.931950,       (4,  3)),
        ('next_size_2',         0.605353,       (4,  2)),
        ('remaining_chunks_7',  0.5,            (5, -1)),
        ('next_size_0',         0.153233,       (4,  0)),
        ('throughput_4',        0.155318,       (2, -4))]
MASK = np.zeros((6, 8))

np.random.seed(RANDOM_SEED)


def main(nn_model, net_traces, output_dir, num_features):

    for i, (f, v, p) in enumerate(FEATURES[:num_features]):
        print('Included:', f)
        MASK[p] = 1
    for i, (f, v, p) in enumerate(FEATURES[num_features:]):
        print('Excluded:', f, f'(default {v})')

    np.set_printoptions(linewidth = 120)
    print(MASK)


    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(net_traces)
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        saver.restore(sess, nn_model)
        print("Testing model restored.")

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        state_tabular_record = list()

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # remove all information from non-included features
            state *= MASK

            for f, v, p in FEATURES[num_features:]:
                state[p] = v

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            tabular_record = str(bit_rate) + ',' + str(reward) + ',' + str(rebuf) + ','
            tabular_record += ','.join(map(str, list(action_prob[0]))) + ','
            tabular_record += str(time_stamp) + ','
            tabular_record += ','.join(map(str, list(np.reshape(state, (S_INFO * S_LEN))))) + '\n'

            state_tabular_record.append(tabular_record)

            s_batch.append(state)

            if end_of_video:
                filename = all_file_names[net_env.trace_idx-1].split('/')[-1]
                output_file = f'{output_dir}/{filename}.csv'

                with open(output_file, 'w') as out:
                    out.write(''.join(state_tabular_record))

                del state_tabular_record[:]

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate execution traces from network information')
    parser.add_argument('k', help='Number of features included', type=int)
    parser.add_argument('nn_model')
    parser.add_argument('net_traces', help='Network state traces', nargs='+')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    main(args.nn_model, args.net_traces, args.output_dir, args.k)
