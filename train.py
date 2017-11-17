#!/usr/bin/python3
#
# Trains a neural network on previously prepared bundles. Saves the model.
#

from common import *

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tflearn
import sklearn as sk
import sklearn.metrics
# from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

log = logging.getLogger("train")

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", type=str, required=True,
                    help="Name of bundle containing the dataset to train on. "
                         "It must be in bundles base dir specified in config "
                         "file.")
parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                    help="Filepath to config file.")
parser.add_argument("--learningrate", type=float, default=0.02,
                    help="Learning rate")
parser.add_argument("--numunits", type=int, default=64,
                    help="Number of LSTM units")
parser.add_argument("--numepochs", type=int, default=2,
                    help="Number of training epochs")
parser.add_argument("--activationfunc", type=str, default="tanh",
                    help="Activation function")
parser.add_argument("--objectivefunc", type=str, default="roc_auc_score",
                    help="Objective function")
args = parser.parse_args()


def load_bundle(bundles_base_dir, bundle_name):
    in_vec_fname = generate_bundle_filename(basename=args.bundle,
                                            is_train=True,
                                            is_input=True)
    out_vec_fname = generate_bundle_filename(basename=args.bundle,
                                             is_train=True,
                                             is_input=False)
    in_vec_path = os.path.join(bundles_base_dir, in_vec_fname) + ".npy"
    out_vec_path = os.path.join(bundles_base_dir, out_vec_fname) + ".npy"

    log.info("Loading from {} and {}.".format(in_vec_path, out_vec_path))

    input_vec = np.load(in_vec_path)
    output_vec = np.load(out_vec_path)
    log.debug("Input vec shape: {}, Output vec shape: {}"
             .format(input_vec.shape, output_vec.shape))

    return input_vec, output_vec


def get_shape(input_vec):
    num_batches = input_vec.shape[0]
    batch_size = input_vec.shape[1]
    embedding_len = input_vec.shape[2]
    log.debug("Num batches = {}; batch size = {}; "
              "embedding len = {}.".format(num_batches, batch_size,
                                          embedding_len))
    return num_batches, batch_size, embedding_len


def make_model(batch_size, embedding_len, activation, n_units, loss, 
        learning_rate):
    num_outputs = 2
    net = tflearn.input_data([None, batch_size, embedding_len])
    net = tflearn.simple_rnn(net,
                             activation=activation,
                             n_units=n_units)
    net = tflearn.dropout(net,
                          keep_prob=0.5)
    net = tflearn.fully_connected(net, num_outputs,
                                  activation="softmax",
                                  name="fc_layer")
    net = tflearn.regression(net,
                             optimizer="adam",
                             loss=loss,
                             learning_rate=learning_rate)
    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=3)
    return model


def generate_run_id(bundle_name):
    run_timestamp = str(datetime.now()).split('.')[0] \
                                       .replace(" ", "_") \
                                       .replace(":", "-")
    return "{}_{}".format(run_timestamp, bundle_name)


def print_metrics(predictions, expectations):
    predicted_class = np.asarray(predictions).astype(float).argmax(1)
    expected_class = expectations.argmax(1)

    results = \
        """
Accuracy: {}
Precision: {}
Recall: {}
F1: {}
Confusion matrix:
{}
        """.format(
            sk.metrics.accuracy_score(expected_class, predicted_class),
            sk.metrics.precision_score(expected_class, predicted_class),
            sk.metrics.recall_score(expected_class, predicted_class),
            sk.metrics.f1_score(expected_class, predicted_class),
            sk.metrics.confusion_matrix(expected_class, predicted_class))

    log.info(results)


def main():
    config = configparser.ConfigParser()
    config.read([args.config])
    init_logger(config)
    bundles_base_dir = config.get("Train", "bundles_base_dir")

    input_vec, output_vec = load_bundle(bundles_base_dir, args.bundle)
    num_batches, batch_size, embedding_len = get_shape(input_vec)

    model = make_model(batch_size, embedding_len,
                       activation=args.activationfunc,
                       n_units=args.numunits,
                       loss=args.objectivefunc,
                       learning_rate=args.learningrate)

    run_id = generate_run_id(args.bundle)
    log.info("Training model...")
    model.fit(input_vec, output_vec, n_epoch=args.numepochs, validation_set=0.3,
              show_metric=True, batch_size=batch_size, run_id=run_id)

    log.info("Predicting across the whole input vector...")
    net_output = model.predict(input_vec)
    print_metrics(predictions=net_output, expectations=output_vec)


if __name__ == "__main__":
    main()
