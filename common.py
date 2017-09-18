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

def init_logger(config):
    log_cfg_path = config.get("Common", "logging_cfg")
    logging.config.fileConfig(log_cfg_path)
