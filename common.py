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

IN_BASE_DIR = "."
OUT_BASE_DIR = "."

BENEVOLENT_FILENAME = "benevolent.txt"
MALEVOLENT_FILENAME = "malevolent.txt"

def init_logger(config):
    log_cfg_path = config.get("Common", "logging_cfg")
    logging.config.fileConfig(log_cfg_path)

def load_operation_dict(filepath):
    result = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                op_name, op_code = line.rstrip().split(":")
                result[op_name] = int(op_code)
    return result