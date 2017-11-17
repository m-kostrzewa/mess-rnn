#!/usr/bin/python3
#
# Trains a neural network on previously prepared bundles. Saves the model.
#

from common import *

import os

import numpy as np
import tensorflow as tf
import tflearn
# from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

log = logging.getLogger("bundle")

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


def main():
    config = configparser.ConfigParser()
    config.read([args.config])
    init_logger(config)
    bundles_base_dir = config.get("Train", "bundles_base_dir")

    in_vec_fname = generate_bundle_filename(basename=args.bundle,
                                            is_train=True,
                                            is_input=True)
    in_vec_path = os.path.join(bundles_base_dir, in_vec_fname) + ".npy"

    out_vec_fname = generate_bundle_filename(basename=args.bundle,
                                             is_train=True,
                                             is_input=False)
    out_vec_path = os.path.join(bundles_base_dir, out_vec_fname) + ".npy"

    log.info("Loading from {} and {}.".format(in_vec_path, out_vec_path))

    input_vec = np.load(in_vec_path)
    output_vec = np.load(out_vec_path)

    num_batches = input_vec.shape[0]
    batch_size = input_vec.shape[1]
    embedding_len = input_vec.shape[2]

    log.info("Loaded training data. Num batches = {}; batch size = {}; "
             "embedding len = {}.".format(num_batches, batch_size,
                                          embedding_len))
    log.info("Input vec shape: {}, Output vec shape: {}"
             .format(input_vec.shape,output_vec.shape))

    num_outputs = 2
    net = tflearn.input_data([None, batch_size, embedding_len])
    net = tflearn.simple_rnn(net, activation=args.activationfunc,
                             n_units=args.numunits)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, num_outputs, activation="softmax",
                                  name="fc_layer")
    net = tflearn.regression(net, optimizer="adam", loss=args.objectivefunc,
                             learning_rate=args.learningrate)
    model = tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=3)

    #run_timestamp = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "-")


    model.fit(input_vec, output_vec, n_epoch=args.numepochs, validation_set=0.3,
              show_metric=True, batch_size=batch_size)#, run_id=run_timestamp)


if __name__ == "__main__":
    main()
