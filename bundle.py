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
from functools import reduce
from collections import namedtuple
import numpy as np

log = logging.getLogger("bundle")
config = configparser.ConfigParser()


def main():
    args = parse_args()
    init(args.config)
    in_subdir = args.input

    in_base_dir = config.get("Workspace", "encoded_base_dir")
    in_abs_dir = os.path.join(in_base_dir, in_subdir)

    found_encodings = find_files_recursive(in_abs_dir, "txt")
    malevolent_encodings = [x for x in found_encodings if TARGET_FILENAME in x]
    benevolent_encodings = [x for x in found_encodings if OTHERS_FILENAME in x]

    bundle(malevolent_encodings, benevolent_encodings, args.name,
           args.batch_size, args.normalize, args.keep_percent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to directory containing encodings that "
                             "are to be bundled into one dataset. Must be "
                             "relative to encoded_base_dir in config.")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of created bundle.")
    parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                        help="Config filepath.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Size of trainable batches for the neural net.")
    parser.add_argument("--max_seq_size", type=int, default=10000,
                        help="Max number of events in a sequence. This may "
                             "cut short any sequences that spam events.")
    # parser.add_argument("--test_set_ratio", type=float, default=0.3,
    #                     help="Ratio of test set size to train set size")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to equalize (clone) malicious batches "
                             "to match benevolent.")
    parser.add_argument("--keep_percent", type=int, default=100,
                        help="Percent of benevolent batches to keep. This "
                             "is useful when there are memory constraints.")
    return parser.parse_args()


def init(config_path):
    config.read([config_path])
    init_logger(config)


def find_encodings(directory, filename):
    encodings = glob.glob("%s/**/%s" % (directory, filename),
                          recursive=True)
    return encodings


def bundle(malevolent_encodings, benevolent_encodings, name, batch_size,
           normalize, keep_percent):
    # TODO: split test and train sets here (before normalization)

    encodings_labels_pairs = [(x, True) for x in benevolent_encodings] + \
                             [(x, False) for x in malevolent_encodings]

    benevolent_batch_num = 0
    malevolent_batch_num = 0
    
    for enc_path, is_benevolent in encodings_labels_pairs:
        st = collect_stats([enc_path], batch_size)
        if is_benevolent:
            benevolent_batch_num += st.batch_num
        else:
            malevolent_batch_num += st.batch_num
        s = "Benevolent" if is_benevolent else "Malevolent"
        log.info("Stats of %s encoding %s: %s" % (s, enc_path, st))

    batch_size = int(batch_size)
    num_batches = benevolent_batch_num + malevolent_batch_num
    embedding_len = calculate_embedding_len(config)

    keep_ratio = (float(keep_percent) / 100.0)
    benevolent_batch_num = int(keep_ratio * benevolent_batch_num)
    log.info("Benevolent keep ratio is %s so will only take %s benevolent "
             "batches." % (keep_ratio, benevolent_batch_num))

    num_batches = benevolent_batch_num + malevolent_batch_num

    if not normalize:
        benevolents_per_malevolent = 1
    if normalize:
        benevolents_per_malevolent = benevolent_batch_num / malevolent_batch_num
        benevolents_per_malevolent = int(benevolents_per_malevolent)
        malevolent_batch_num = malevolent_batch_num * \
                                   benevolents_per_malevolent
        num_batches = benevolent_batch_num + malevolent_batch_num
        log.info("Normalizing: new num batches: %s (malicious: %s)" %
                 (num_batches, malevolent_batch_num))

    batches = batch_generator(encodings_labels_pairs, batch_size,
                              benevolents_per_malevolent, benevolent_batch_num)

    input_vec, labels_vec = allocate_vectors(num_batches, batch_size,
                                             embedding_len)
    input_vec, labels_vec = populate_vectors(input_vec, labels_vec, batches)

    log.info("Final shapes: input vec: {}, labels vec: {}"
             .format(input_vec.shape, labels_vec.shape))

    out_base_dir = config.get("Workspace", "bundles_base_dir")
    if not os.path.exists(out_base_dir):
        os.makedirs(out_base_dir)

    in_vec_filename = generate_bundle_filename(basename=name,
                                               is_train=True,
                                               is_input=True)
    in_vec_filepath = os.path.join(out_base_dir, in_vec_filename)
    save_to_file(input_vec, in_vec_filepath)

    labels_filename = generate_bundle_filename(basename=name,
                                               is_train=True,
                                               is_input=False)
    labels_vec_filepath = os.path.join(out_base_dir, labels_filename)
    save_to_file(labels_vec, labels_vec_filepath)

    log.info("Saved to: %s, %s" % (in_vec_filepath, labels_vec_filepath))
    return in_vec_filepath, labels_vec_filepath


def collect_stats(files, batch_size):
    proc_num = 0
    total_len = 0
    batch_num = 0
    for path in files:
        with open(path) as f:
            for line in f:
                if ":" in line:
                    line = line.split(":")[-1]
                proc_num += 1
                this_proc_len = len(line.split(","))
                total_len += this_proc_len
                batch_num += int(this_proc_len / batch_size)
    Stats = namedtuple("Stats", ["proc_num", "total_len", "batch_num"])
    return Stats(proc_num=proc_num, total_len=total_len, batch_num=batch_num)


def calculate_embedding_len(config):
    p = config.get("Workspace", "dictionary_path")
    opcodes_dict = load_operation_dict(p)
    embedding_len = max(opcodes_dict.values()) + 1
    log.info("Embedding len: %s" % embedding_len)
    return embedding_len


def allocate_vectors(num_batches, batch_size, embedding_len):
    sizeof_f32 = 4
    size_in = num_batches * batch_size * embedding_len * sizeof_f32 / 1024**2
    size_out = num_batches * 2 * sizeof_f32 / 1024**2
    size_total = size_in + size_out
    log.info("Estimated memory required: %i MB" % size_total)
    input_vec = np.empty((num_batches, batch_size, embedding_len),
                         dtype=np.float32)
    labels_vec = np.empty((num_batches, 2),
                          dtype=np.float32)
    log.info("Input vec: {}; output vec: {}".format(input_vec.shape,
                                                    labels_vec.shape))
    return input_vec, labels_vec


def batch_generator(encodings_labels_pairs, batch_size,
                    repeats_of_malevolents, benevolents_cutoff):

    event_sequences = event_sequence_generator(encodings_labels_pairs)
    num_events = 0
    benevolent_batch_counter = 0
    current_batch = []
    for is_benevolent, seq in event_sequences:
        if num_events != 0:
            # Last batch was not completed (not enough events). Instead
            # of padding let's just drop it.
            num_events = 0
            current_batch = []
        for event_opcode in seq:
            if is_benevolent and benevolent_batch_counter >= benevolents_cutoff:
                # benevolents keep ratio: don't add any benevolent batches
                # if there is enough of them.
                continue
            current_batch.append(event_opcode)
            num_events += 1
            if num_events == batch_size:
                if not is_benevolent:
                    # normalization - clone malicious batches if needed
                    for j in range(repeats_of_malevolents):
                        yield is_benevolent, current_batch
                else:
                    benevolent_batch_counter += 1
                    yield is_benevolent, current_batch
                num_events = 0
                current_batch = []


def event_sequence_generator(encodings_labels_pairs):
    for to_load in encodings_labels_pairs:
        with open(to_load[0]) as f:
            for line in f:
                if ":" in line:
                    line = line.split(":")[-1]
                yield to_load[1], line.split(",")


def populate_vectors(input_vec, labels_vec, batches):
    batch_size = input_vec.shape[1]
    embedding_len = input_vec.shape[2]
    for batch_idx, (is_benevolent, batch) in enumerate(batches):
        tmp_np_batch = np.empty((batch_size, embedding_len))
        for event_idx, event_opcode in enumerate(batch):
            event_embedding = embed_event(int(event_opcode), embedding_len)
            tmp_np_batch[event_idx] = event_embedding
        input_vec[batch_idx] = tmp_np_batch
        labels_vec[batch_idx] = embed_output(is_benevolent)
    log.debug("Input vec:\n%s" % input_vec)
    log.debug("Output vec:\n%s" % labels_vec)
    return input_vec, labels_vec


def embed_event(opcode, embedding_len):
    embedding = np.zeros(embedding_len)
    if opcode != -1:
        # -1 means opcode not present
        embedding[opcode] = 1
    return embedding


def embed_output(is_benevolent):
    return np.array([1, 0]) if is_benevolent else np.array([0, 1])


def save_to_file(matrix_to_save, filepath):
    out_filepath = "{}.npy".format(filepath)
    np.save(out_filepath, matrix_to_save)


if __name__ == "__main__":
    main()
