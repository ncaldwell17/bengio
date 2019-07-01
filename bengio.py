import tensorflow as tf
import math
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from collections import Counter

# helper functions


# reads the file as a list of words
def read_file(path_to_file):
    with open(path_to_file, "r") as text:
        ls_words = text.read().replace("\n", "<eos>").split()
    # Clean the vocab from random characters within the corpora
    regex = re.compile(r'[.a-zA-Z0-9]')
    if a3 == 'wiki':
        return [i.lower() for i in ls_words if (regex.search(i) or i == '<eos>')]
    else:
        return [i for i in ls_words if (regex.search(i) or i == '<eos>')]


# makes batches of data for later use
def make_batches(data, batch_size, window_size):
    x_data = []
    y_data = []
    for i in range(len(data)):
        if i > window_size - 1:
            x_data.append(data[i - window_size:i])
            y_data.append(data[i])
    batches = int(len(x_data) / batch_size)
    batch_out = list()
    for i in range(batches):
        # For each batch
        start_i = batch_size * i
        end_i = start_i + batch_size
        x_values = x_data[start_i:end_i]
        y_values = y_data[start_i:end_i]
        batch_out.append([x_values, y_values])
    return batch_out


# brown corpus is huge, splits according to Bengio layout in paper
def split_brown():
    with open('data/brown.txt') as file:
        text_list = file.read().split()
    training = ' '.join(text_list[:800000])
    training_file = open("data/brown.train.txt", "w")
    training_file.write(training)
    training_file.close()

    validation = ' '.join(text_list[800000:1000000])
    validation_file = open("data/brown.valid.txt", "w")
    validation_file.write(validation)
    validation_file.close()

    testing = ' '.join(text_list[1000000:])
    testing_file = open("data/brown.test.txt", "w")
    testing_file.write(testing)
    testing_file.close()


# plots the learning curve in plt
def plot_learning(accuracy, cost):
    loss = [1 - x for x in accuracy]
    figure = plt.figure(figsize=(10, 6))
    x = np.arange(0, np.shape(cost)[0])
    plt.subplot(2, 1, 1)
    plt.plot(x, cost, color='red')
    plt.title('Validation Cost')
    plt.subplot(2, 1, 2)
    plt.plot(x, accuracy, c='b')
    plt.title('Validation Accuracy')
    plt.show()


class Preprocessor:

    def __init__(self, path):
        raw_list = read_file(path)
        top_words = Counter(raw_list).most_common()
        words = [word[0] for word in top_words if word[1] >= 3]
        if '<unk>' in words:
            words.remove('<unk>')
        self.word_dict = {'<unk>': 0}
        for i in range(1, len(words)):
            self.word_dict[words[i]] = i
        self.vocab_size = len(self.word_dict)
        self.word_dict_reverse = dict(zip(self.word_dict.values(), self.word_dict.keys()))
        self.text_as_index = []
        for word in words:
            idx = 0
            if word in self.word_dict:
                idx = self.word_dict[word]
            self.text_as_index.append(idx)

    def generate_data(self, path):
        words = read_file(path)
        text_as_index = []
        for word in words:
            idx = 0
            if word in self.word_dict:
                idx = self.word_dict[word]
            text_as_index.append(idx)
        return text_as_index


class BengioModel:

    def __init__(self):
        self.batch_size = 256
        self.embedding_size = user_input['embedding_size']
        self.window_size = user_input['window_size']
        self.hidden_units = user_input['hidden_units']

    def train_model(self, training_data, validation_data, num_epochs=5):
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            device = '/gpu:0'
            print('Currently using GPU device to run code')
        # need to figure out how to run this on a TPU
        else:
            device = '/cpu:0'
            print('There is no GPU available, using CPU device to run code')
        with tf.device(device):
            # all code that I want to run on GPU insert here
            self.x_input = tf.placeholder(tf.int64, [None, self.window_size])
            self.y_input = tf.placeholder(tf.int64, [None])

            z = self.embedding_size * self.window_size

            # hidden layer biases
            d = tf.Variable(tf.random_uniform([self.hidden_units]))
            # output biases
            b = tf.Variable(tf.random_uniform([vocab_size]))

            # weights
            # C matrix function
            word_embeddings = tf.Variable(tf.random_uniform([vocab_size,
                                                             self.embedding_size],
                                                            -1.0,
                                                            1.0))
            flattened_exes = tf.layers.flatten(self.x_input)
            lookup = tf.nn.embedding_lookup(word_embeddings, flattened_exes)
            xt = tf.reshape(lookup, [self.batch_size, z])

            # H Weight
            H = tf.Variable(tf.truncated_normal([z, self.hidden_units],
                                                stddev=1.0 / math.sqrt(z)))
            # W Weight
            W = tf.Variable(tf.truncated_normal([z, vocab_size],
                                                stddev=1.0 / math.sqrt(z)))
            # U Weight
            U = tf.Variable(tf.truncated_normal([self.hidden_units, vocab_size],
                                                stddev=1.0 / vocab_size))

            # hidden layers
            tanh = tf.nn.tanh(tf.nn.bias_add(tf.matmul(xt, H), d))
            y = tf.nn.bias_add(tf.matmul(xt, W), b) + tf.matmul(tanh, U)

            # softmax
            y_probability_distribution = tf.nn.softmax(y)
            y_ideal = tf.argmax(y_probability_distribution, axis=1)
            # produces labels to use in the softmax_cross_entropy_with_logits
            y_labels = tf.one_hot(self.y_input, vocab_size)

            self.ce_result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labels,
                                                                                       logits=y_probability_distribution))

            # stochastic gradient descent optimizer
            learning_rate = 0.001
            beta1 = 0.9
            beta2 = 0.999
            adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(self.ce_result)
            ra = tf.equal(y_ideal, self.y_input)
            self.accuracy = tf.reduce_mean(tf.cast(ra, tf.float32))

            self.session = tf.Session()
            self.session.run(tf.initializers.global_variables())
            saver = tf.train.Saver()

            print('training beginning . . . . .')
            global training_accuracy, training_cost
            for i in range(num_epochs):
                batches = make_batches(training_data,
                                       self.batch_size,
                                       self.window_size)
                total_batches = len(batches)
                batch_count = 0
                last_complete = 0
                num_messages = 10  # the number of  printouts  per  epoch
                for batch in batches:
                    batch_count += 1
                    x_batch = batch[0]
                    y_batch = batch[1]
                    feed_dict_train = {self.x_input: x_batch,
                                       self.y_input: y_batch}
                    self.session.run(adam, feed_dict=feed_dict_train)
                    completion = 100 * batch_count / total_batches
                    if batch_count % (int(total_batches / num_messages)) == 0:
                        print('Epoch #%2d-   Batch #%5d:   %4.2f %% completed.' % (i + 1, batch_count, completion))
                        a_t, c_t = self.test(training_data)
                        a, c = self.test(validation_data)
                        training_accuracy.append(a)
                        training_cost.append(c)

                        if sum(training_cost[-4:]) > sum(training_cost[-8:-4]):
                            patience = patience - 1
                        else:
                            patience = 2

                        if patience == 0:
                            print("Cost Too High, Early Stop Activated")
                            save_path = saver.save(self.session, "../models/" + a2 + '_' + a3 + ".ckpt")
                            print("Model saved in path: %s" % save_path)
                            return

        print("Training is finished")
        save_path = saver.save(self.session, "../models/" + a2 + '_' + a3 + ".ckpt")
        print("Model saved in path: %s" % save_path)
        return

    def restore_model(self, path):
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            device = '/gpu:0'
            print('Currently using GPU device to run code')
        # need to figure out how to run this on a TPU
        else:
            device = '/cpu:0'
            print('There is no GPU available, using CPU device to run code')
        with tf.device(device):
            # C/P everything from def train_model above until SGD optimizer
            # all code that I want to run on GPU insert here
            self.x_input = tf.placeholder(tf.int64, [None, self.window_size])
            self.y_input = tf.placeholder(tf.int64, [None])

            z = self.embedding_size * self.window_size

            # hidden layer biases
            d = tf.Variable(tf.random_uniform([self.hidden_units]))
            # output biases
            b = tf.Variable(tf.random_uniform([vocab_size]))

            # weights
            # C matrix function
            word_embeddings = tf.Variable(tf.random_uniform([vocab_size,
                                                             self.embedding_size],
                                                            -1.0,
                                                            1.0))
            flattened_exes = tf.layers.flatten(self.x_input)
            lookup = tf.nn.embedding_lookup(word_embeddings, flattened_exes)
            xt = tf.reshape(lookup, [self.batch_size, z])

            # H Weight
            H = tf.Variable(tf.truncated_normal([z, self.hidden_units],
                                                stddev=1.0 / math.sqrt(z)))
            # W Weight
            W = tf.Variable(tf.truncated_normal([z, vocab_size],
                                                stddev=1.0 / math.sqrt(z)))
            # U Weight
            U = tf.Variable(tf.truncated_normal([self.hidden_units, vocab_size],
                                                stddev=1.0 / vocab_size))

            # hidden layers
            tanh = tf.nn.tanh(tf.nn.bias_add(tf.matmul(xt, H), d))
            y = tf.nn.bias_add(tf.matmul(xt, W), b) + tf.matmul(tanh, U)

            # softmax
            y_probability_distribution = tf.nn.softmax(y)
            y_ideal = tf.argmax(y_probability_distribution, axis=1)
            # produces labels to use in the softmax_cross_entropy_with_logits
            y_labels = tf.one_hot(self.y_input, vocab_size)

            self.ce_result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_labels,
                                                                                            y_probability_distribution))

            # stochastic gradient descent optimizer
            learning_rate = 0.001
            beta1 = 0.9
            beta2 = 0.999
            adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2).maximize(self.ce_result)
            ra = tf.equal(y_ideal, self.y_input)
            self.accuracy = tf.reduce_mean(tf.cast(ra, tf.float32))

            self.session = tf.Session()
            self.session.run(tf.global_variables_initalizer())
            saver = tf.train.Saver()

            with tf.Session() as session:
                # returns variables from the disk
                saver.restore(session, path)
                print("The Model has been restored from disk")
                test_batches = make_batches(test_data, self.batch_size, self.window_size)
                cost, accuracy = [], []
                for batch in test_batches:
                    feed_dict_test = {self.x_input: batch[0],
                                      self.y_input: batch[1]}
                    accuracy.append(session.run(self.accuracy,
                                                feed_dict=feed_dict_test))
                    cost.append(session.run(self.ce_result,
                                            feed_dict=feed_dict_test))
                average_accuracy = sum(accuracy) / float(len(accuracy))
                average_cost = sum(cost) / float(len(cost))
                print("   Accuracy on test-set:   %4.2f %% \n" % (average_accuracy * 100),
                      "   Cost on test-set:       %4.2f \n" % average_cost,
                      "   Perplexity on test-set:       %4.2f \n" % np.exp(average_cost))

    def test(self, test_data):
        test_batches = make_batches(test_data, self.batch_size, self.window_size)
        cost, accuracy = [], []
        for batch in test_batches:
            feed_dict_test = {self.x_input: batch[0],
                              self.y_input: batch[1]}
            accuracy.append(self.session.run(self.accuracy, feed_dict=feed_dict_test))
            cost.append(self.session.run(self.ce_result, feed_dict=feed_dict_test))
        average_accuracy = sum(accuracy) / float(len(accuracy))
        average_cost = sum(cost) / float(len(cost))
        print("   Accuracy on valid-set:   %4.2f %%" % (average_accuracy * 100),
              "   Cost on valid-set:       %4.2f \n" % average_cost)
        return average_accuracy, average_cost


# key variables
a1 = sys.argv[1]
a2 = sys.argv[2]
a3 = sys.argv[3]

user_inputs = {'MLP1': {'window_size': 5, 'hidden_units': 50, 'embedding_size': 60, 'direct': True, 'mix': False},
           'MLP5': {'window_size': 5, 'hidden_units': 0, 'embedding_size': 30, 'direct': True, 'mix': False}}

corpora = ['wiki', 'brown']


if __name__ == "__main__":
    if a1 not in ['train', 'load'] or a2 not in user_inputs or a3 not in corpora:
        print("please enter in a valid input")
        sys.exit()
    elif a1 == 'train':
        if a3 == 'wiki':
            train_path = "data/wiki.train.txt"
            validation_path = "data/wiki.valid.txt"
            test_path = "data/wiki.test.txt"
        elif a3 == 'brown':
            split_brown()
            train_path = "data/brown.train.txt"
            validation_path = "data/brown.valid.txt"
            test_path = "data/brown.test.txt"

        user_input = user_inputs[a2]
        corpus = Preprocessor(train_path)
        vocab_size = corpus.vocab_size
        train_data = corpus.generate_data(train_path)
        validate_data = corpus.generate_data(validation_path)
        test_data = corpus.generate_data(test_path)
        model = BengioModel()
        acc_hist_train, cost_hist_train = [.1] * 10, [7] * 10
        model.train_model(train_data, validate_data)
        plot_learning(training_accuracy[10:], training_cost[10:])

    elif a1 == 'load':
        if a3 == 'brown':
            split_brown()
            train_path = "data/brown.train.txt"
            path_to_validation_file = "data/brown.valid.txt"
            path_to_testing_file = "data/brown.test.txt"
        elif a3 == 'wiki':
            train_path = "data/wiki.train.txt"
            validation_path = "data/wiki.valid.txt"
            test_path = "data/wiki.test.txt"

        configuration = configs[a2]
        corpus = Preprocessor(train_path)
        vocab_size = corpus.vocab_size
        test_data = corpus.generate_data(test_path)
        model = BengioModel()
        model.restore_model('../models/' + a2 + '_' + a3 + '.ckpt')



