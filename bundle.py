#!/usr/bin/python3
#
# Takes encoded history of each process in specified subdirectory and outputs
# numpy datasets easily digestible in Tensorflow. Also, splits those datasets
# into train and tests subsets.
#

from common import *

import os
import re
import glob
from collections import namedtuple
import numpy as np

log = logging.getLogger("bundle")

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True,
                    help="Directory containing raw input samples relative to "
                         "input base dir specified in config file. It is "
                         "also the name of output directory under base output "
                         "dir, also specified in config file.")
parser.add_argument("--config", type=str,
                    help="Filepath to config file.")
parser.add_argument("--batch_size", type=int, default=50,
                    help="Size of batch")
parser.add_argument("--max_seq_size", type=int, default=10000,
                    help="Max number of events in a sequence")
# parser.add_argument("--test_set_ratio", type=float, default=0.3,
#                     help="Ratio of test set size to train set size")
# parser.add_argument("--normalize", type=int, default=10000,
#                     help="Whether to equalize the number of batches of each "
#                          "type")
parser.add_argument("--name", type=str, required=True,
                    help="Name of created bundle")
args = parser.parse_args()


IN_BASE_DIR = "."
OUT_BASE_DIR = "."


def collect_stats(files_glob):
    proc_num = 0
    total_len = 0
    batch_num = 0
    for path in files_glob:
        with open(path) as f:
            for line in f:
                proc_num += 1
                this_proc_len = len(line.split(","))
                total_len += this_proc_len
                batch_num += int(this_proc_len / args.batch_size)
    Stats = namedtuple("Stats", ["proc_num", "total_len", "batch_num"])
    return Stats(proc_num=proc_num, total_len=total_len, batch_num=batch_num)


def save_to_file(matrix_to_save, filename):
    out_filename = "{}.npy".format(filename)
    out_path = os.path.join(OUT_BASE_DIR, args.dir, out_filename)
    np.save(out_path, matrix_to_save)



def main():
    global IN_BASE_DIR, OUT_BASE_DIR

    config = configparser.ConfigParser()
    config.read([args.config])
    init_logger(config)
    IN_BASE_DIR = config.get("Bundle", "input_base_dir")
    OUT_BASE_DIR = config.get("Bundle", "output_base_dir")







    malovelent_encodings = glob.glob("%s/%s/**/%s" % (IN_BASE_DIR, args.dir,
                                                      MALEVOLENT_FILENAME),
                                recursive=True)
    malevolent_stats = collect_stats(malovelent_encodings)

    benevolent_encodings = glob.glob("%s/%s/**/%s" % (IN_BASE_DIR, args.dir,
                                                      BENEVOLENT_FILENAME),
                                recursive=True)
    benevolent_stats = collect_stats(benevolent_encodings)

    log.info("Found malevolent sample stats: {}".format(malevolent_stats))
    log.info("Found benevolent sample stats: {}".format(benevolent_stats))


    # TODO: split test and train sets here (before normalization)
    # TODO: normalize num of samples in each step

    p = config.get("Common", "dictionary_path")
    opcodes_dict = load_operation_dict(p)
    embedding_len = max(opcodes_dict.values()) + 1
    log.info("Embedding len: %s" % embedding_len)

    batch_size = int(args.batch_size)
    num_batches = malevolent_stats.batch_num + benevolent_stats.batch_num

    input_vec = np.empty((num_batches, batch_size, embedding_len))
    output_vec = np.empty((num_batches, 2))
    log.info("Input vec: {}; output vec: {}".format(input_vec.shape,
                                                    output_vec.shape))

    def event_sequence_generator(files_to_load):
        for to_load in files_to_load:
            with open(to_load) as f:
                for line in f:
                    yield to_load.endswith(BENEVOLENT_FILENAME), line.split(",")

    def batch_generator(files_to_load, batch_size):
        event_sequences = event_sequence_generator(files_to_load)
        i = 0
        current_batch = []
        for is_benevolent, seq in event_sequences:
            if i != 0:
                # Last batch was not completed (not enough events). Instead
                # of padding let's just drop it.
                i = 0
                current_batch = []
            for event_opcode in seq:
                current_batch.append(event_opcode)
                i += 1
                if i == batch_size:
                    yield is_benevolent, current_batch
                    i = 0
                    current_batch = []

    def embed_event(opcode, embedding_len):
        embedding = np.zeros(embedding_len)
        embedding[opcode] = 1
        return embedding

    def embed_output(is_benevolent):
        return np.array([1, 0]) if is_benevolent else np.array([0, 1])

    files_to_load = malovelent_encodings + benevolent_encodings
    batches = batch_generator(files_to_load, batch_size)
    for batch_idx, (is_benevolent, batch) in enumerate(batches):

        tmp_np_batch = np.empty((batch_size, embedding_len))
        for event_idx, event_opcode in enumerate(batch):
            event_embedding = embed_event(int(event_opcode), embedding_len)
            tmp_np_batch[event_idx] = event_embedding

        input_vec[batch_idx] = tmp_np_batch
        output_vec[batch_idx] = embed_output(is_benevolent)

    log.debug("Input vec:\n%s" % input_vec)
    log.debug("Output vec:\n%s" % output_vec)

    if not os.path.exists(OUT_BASE_DIR):
        os.makedirs(OUT_BASE_DIR)

    input_vec_filename = "{}_train_input_vec".format(args.name)
    save_to_file(input_vec, os.path.join(OUT_BASE_DIR, input_vec_filename))

    output_vec_filename = "{}_train_output_vec".format(args.name)
    save_to_file(output_vec, os.path.join(OUT_BASE_DIR, output_vec_filename))


if __name__ == "__main__":
    main()
