#!/usr/bin/python3
#
# Invokes tensorflow neural network. This script can be used to train a network,
# load weights from previously trained network or just test it on some data.
#

from common import *

import os
import glob
from collections import namedtuple
import pickle

import numpy as np
import tensorflow as tf
import tflearn
import sklearn as sk
import sklearn.metrics
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

log = logging.getLogger("rnn")
config = configparser.ConfigParser()


def main():
    args = parse_args()
    init(args.config)
    bundles_base_dir = config.get("Workspace", "bundles_base_dir")
    weights_base_dir = config.get("Workspace", "weights_base_dir")
    hparams_dir = config.get("Workspace", "hyperparams_base_dir")

    bundle_name = args.bundle
    input_vec, output_vec = load_bundle(bundles_base_dir, bundle_name)
    tensorboard_dir = config.get("Workspace", "tensorboard_dir")

    if args.model is None:
        model_name = generate_model_name(bundle_name)
    else:
        model_name = args.model
    log.info("Model name of this run: %s" % model_name)

    if args.load:
        hyperparams = load_hyperparams(model_name, hparams_dir)
    else:
        hyperparams = get_hyperparams(args)
        save_hyperparams(hyperparams, model_name, hparams_dir)
    log.info(hyperparams)

    data_shape = get_shape(input_vec)
    model = make_model(data_shape, hyperparams, tensorboard_dir, model_name,
                       args.use_lstm)

    if args.load:
        model = load_weights(model, model_name, weights_base_dir)

    if args.train:
        train(model, input_vec, output_vec, model_name, data_shape,
              hyperparams)
        save_weights(model, model_name, weights_base_dir)

    log.info("Predicting across the whole input vector...")
    net_output = model.predict(input_vec)
    print_metrics(predictions=net_output, expectations=output_vec)
    return net_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=str, required=True,
                        help="Name of bundle containing the dataset to train "
                             "or test, depending on --train flag."
                             "It must be in bundles base dir specified in config "
                             "file.")
    parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                        help="Filepath to config file.")

    parser.add_argument("--train", action="store_true",
                        help="If true, will train the network and test it on "
                             "the whole dataset at the end. If --pretrained "
                             "is not specified, will create a new pretrained. "
                             "If --pretrained is specified, will start "
                             "training using that pretrained model."
                             "If false, will only test on the whole dataset, "
                             "but pretrained MUST be specified.")
    parser.add_argument("--model", type=str, default=None,
                        help="Name of trained model to use. Must be relative "
                             "to best_checkpoint_dir in config. If not "
                             "specified, a new name will be generated.")
    parser.add_argument("--load", action="store_true",
                        help="Whether to load model hyperparameters and "
                             "weights. If false, will get hyperparameters "
                             "based on other arguments.")

    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to useLSTM instead of RNN layer.")

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


def init(config_path):
    config.read([config_path])
    init_logger(config)


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


def get_tflearn_meta(config):
    TFMeta = namedtuple("TFMeta", ["tensorboard_dir",
                                   "best_checkpoint_dir"])
    t = TFMeta(tensorboard_dir=config.get("Workspace", "tensorboard_dir"),
               best_checkpoint_dir=config.get("Workspace",
                                              "best_checkpoint_dir"))
    log.debug(t)
    return t


def generate_model_name(bundle_name):
    run_timestamp = timestamp()
    return "{}_{}".format(run_timestamp, bundle_name)


def get_hyperparams(args):
    h = Hyperparams(num_epochs=args.numepochs,
                    learning_rate=args.learningrate,
                    n_units=args.numunits,
                    activation=args.activationfunc,
                    loss=args.objectivefunc)
    log.debug(h)
    return h


def load_hyperparams(model_name, hparams_dir):
    hyperparams_path = os.path.join(hparams_dir, model_name)
    log.info("Loading hyperparams from %s" % hyperparams_path)
    with open(hyperparams_path, "rb") as f:
        hyperparams = pickle.load(f)
    return hyperparams


def save_hyperparams(hyperparams, model_name, hparams_dir):
    hyperparams_path = os.path.join(hparams_dir, model_name)
    log.info("Saving hyperparams to %s" % hyperparams_path)
    if not os.path.exists(hparams_dir):
        os.makedirs(hparams_dir)
    with open(hyperparams_path, "wb") as f:
        pickle.dump(hyperparams, f)


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


def make_model(data_shape, hyperparams, tensorboard_dir, model_name,
               use_lstm=False):
    num_outputs = 2
    net = tflearn.input_data([None,
                              data_shape.batch_size,
                              data_shape.embedding_len])
    if not use_lstm:
        net = tflearn.simple_rnn(net,
                                 activation=hyperparams.activation,
                                 n_units=hyperparams.n_units)
    else:
        net = tflearn.lstm(net,
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
                        tensorboard_dir=tensorboard_dir)
    return model


def load_weights(model, model_name, weights_base_dir):
    weights_path = get_model_path(weights_base_dir, model_name)
    log.info("Loading weights from %s" % weights_path)
    model.load(weights_path)
    return model


def save_weights(model, model_name, weights_base_dir):
    model_path = get_model_path(weights_base_dir, model_name)
    log.info("Saving weights to %s" % model_path)
    model.save(model_path)


def get_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)


def train(model, input_vec, output_vec, model_name, data_shape, hyperparams):
    log.info("Training model...")
    model.fit(input_vec, output_vec,
              n_epoch=hyperparams.num_epochs,
              validation_set=0.3,
              show_metric=True,
              batch_size=data_shape.batch_size,
              run_id=model_name)


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


if __name__ == "__main__":
    main()
