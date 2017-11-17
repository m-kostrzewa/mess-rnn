#!/usr/bin/python3
#
# Trains a neural network on previously prepared bundles. Saves the model.
#

from common import *

import os
from datetime import datetime
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tflearn
import sklearn as sk
import sklearn.metrics
# from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

log = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=str, required=True,
                        help="Name of bundle containing the dataset to train on. "
                            "It must be in bundles base dir specified in config "
                            "file.")
    parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                        help="Filepath to config file.")

    parser.add_argument("--numepochs", type=int, default=2,
                        help="Number of training epochs")

    parser.add_argument("--learningrate", type=float, default=0.02,
                        help="Learning rate")
    parser.add_argument("--numunits", type=int, default=64,
                        help="Number of LSTM units")
    parser.add_argument("--activationfunc", type=str, default="tanh",
                        help="Activation function")
    parser.add_argument("--objectivefunc", type=str, default="roc_auc_score",
                        help="Objective function")
    args = parser.parse_args()
    return args


def get_hyperparams(args):
    Hyperparams = namedtuple("Hyperparams", ["num_epochs",
                                             "learning_rate",
                                             "n_units",
                                             "activation",
                                             "loss"])
    h = Hyperparams(num_epochs=args.numepochs,
                    learning_rate=args.learningrate,
                    n_units=args.numunits,
                    activation=args.activationfunc,
                    loss=args.objectivefunc)
    log.debug(h)
    return h


def get_tflearn_meta(config):
    TFMeta = namedtuple("TFMeta", ["tensorboard_dir",
                                   "best_checkpoint_path"])
    t = TFMeta(tensorboard_dir=config.get("Train", "tensorboard_dir"),
               best_checkpoint_path=config.get("Train", "best_checkpoint_path"))
    log.debug(t)
    return t


def load_bundle(bundles_base_dir, bundle_name):
    in_vec_fname = generate_bundle_filename(basename=bundle_name,
                                            is_train=True,
                                            is_input=True)
    out_vec_fname = generate_bundle_filename(basename=bundle_name,
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
    Shape = namedtuple("Shape", ["num_batches",
                                 "batch_size",
                                 "embedding_len"])
    s = Shape(num_batches=num_batches,
                 batch_size=batch_size,
                 embedding_len=embedding_len)
    log.debug(s)
    return s


def make_model(data_shape, hyperparams, tf_meta):
    num_outputs = 2
    net = tflearn.input_data([None,
                              data_shape.batch_size,
                              data_shape.embedding_len])
    net = tflearn.simple_rnn(net,
                             activation=hyperparams.activation,
                             n_units=hyperparams.n_units)
    net = tflearn.dropout(net,
                          keep_prob=0.5)
    net = tflearn.fully_connected(net, num_outputs,
                                  activation="softmax",
                                  name="fc_layer")
    net = tflearn.regression(net,
                             optimizer="adam",
                             loss=hyperparams.loss,
                             learning_rate=hyperparams.learning_rate)
    model = tflearn.DNN(net,
                        clip_gradients=5.0,
                        tensorboard_verbose=3,
                        tensorboard_dir=tf_meta.tensorboard_dir,
                        best_checkpoint_path=tf_meta.best_checkpoint_path)
    return model


def generate_run_id(bundle_name):
    run_timestamp = str(datetime.now()).split('.')[0] \
                                       .replace(" ", "_") \
                                       .replace(":", "-")
    return "{}_{}".format(run_timestamp, bundle_name)


def fit_model(model, input_vec, output_vec, bundle, data_shape, hyperparams):
    log.info("Training model...")
    run_id = generate_run_id(bundle)
    model.fit(input_vec, output_vec,
              n_epoch=hyperparams.num_epochs,
              validation_set=0.3,
              show_metric=True,
              batch_size=data_shape.batch_size,
              run_id=run_id)


def print_metrics(predictions, expectations):
    pred_class = np.asarray(predictions).astype(float).argmax(1)
    exp_class = expectations.argmax(1)
    results_string = \
        """
Accuracy: {}
Precision: {}
Recall: {}
F1: {}
Confusion matrix:
{}
        """.format(sk.metrics.accuracy_score(exp_class, pred_class),
                   sk.metrics.precision_score(exp_class, pred_class),
                   sk.metrics.recall_score(exp_class, pred_class),
                   sk.metrics.f1_score(exp_class, pred_class),
                   sk.metrics.confusion_matrix(exp_class, pred_class))
    log.info(results_string)


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read([args.config])
    init_logger(config)
    bundles_base_dir = config.get("Train", "bundles_base_dir")

    bundle = args.bundle
    input_vec, output_vec = load_bundle(bundles_base_dir, bundle)
    data_shape = get_shape(input_vec)

    hyperparams = get_hyperparams(args)
    tf_meta = get_tflearn_meta(config)
    model = make_model(data_shape, hyperparams, tf_meta)
    fit_model(model, input_vec, output_vec, bundle, data_shape, hyperparams)

    log.info("Predicting across the whole input vector...")
    net_output = model.predict(input_vec)
    print_metrics(predictions=net_output, expectations=output_vec)


if __name__ == "__main__":
    main()
