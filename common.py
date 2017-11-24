#!/usr/bin/python3

import configparser
import argparse
import glob
import time
import queue
import threading
import logging
import logging.config
import os.path

TARGET_PROCESS_FILENAME = "target.txt"
OTHER_PROCESSES_FILENAME = "others.txt"


def init_logger(config):
    log_cfg_path = config.get("Common", "logging_cfg")
    logging.config.fileConfig(log_cfg_path)


def find_files_recursive(directory, filetype):
    return glob.glob("%s/*.%s" % (directory, filetype), recursive=True)


def read_all_queue(output_queue):
    buf = []
    while True:
        try:
            item = output_queue.get_nowait()
            buf.append(item)
        except queue.Empty:
            break
    return buf


def load_operation_dict(filepath):
    result = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                op_name, op_code = line.rstrip().split(":")
                result[op_name] = int(op_code)
    return result


def generate_bundle_filename(basename, is_train, is_input):
    return "{}_{}_{}_vec".format(basename,
                                 ("train" if is_train else "test"),
                                 ("input" if is_input else "output"))
