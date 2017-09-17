#!/usr/bin/python3

import xmlrpc.client
import configparser
import argparse
import glob
import time
import queue
import threading
import logging
import logging.config
import os
import urllib.request
import re

log = logging.getLogger("analyze")

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True,
                    help="Directory containing raw input samples relative to "
                         "input base dir specified in config file. It is "
                         "also the name of output directory under base output "
                         "dir, also specified in config file.")
parser.add_argument("--config", type=str,
                    help="Filepath to config file.")
args = parser.parse_args()


class MessWorker(threading.Thread):
    def __init__(self, id, queue, mess_client, toolkit, output_base_dir,
                 sleep_time_minutes, results_url, vm_name, snapshot_name):
        super().__init__()
        self.id = id
        self.queue = queue
        self.mess = mess_client
        self.toolkit = toolkit
        self.output_base_dir = output_base_dir
        self.sleep_time_minutes = sleep_time_minutes
        self.results_url = results_url
        self.vm_name = vm_name
        self.snapshot_name = snapshot_name
        log.info("Starting MessWorker thread %s" % self.id)
        log.debug("MessWorker %s - vm_name=%s, snapshot_name=%s" %
                  (self.id, self.vm_name, self.snapshot_name))

    def run(self):
        while True:
            sample = self.queue.get()
            if sample is None:
                log.warn("MessWorker %s - no next sample, breaking" % self.id)
                break
            self.analyze(sample)
            self.queue.task_done()
            log.info("MessWorker %s - job completed" % self.id)

    def analyze(self, sample):
        assert(self.mess.get_vm_state(self.vm_name) == "READY")

        log.info("MessWorker %s - starting analysis of %s." %
                 (self.id, sample.path))
        with open(self.toolkit, "rb") as f:
            toolkit_data = xmlrpc.client.Binary(f.read())
        self.mess.start_analysis(self.vm_name, sample.target_name,
            sample.command.split("%"), sample.load(), toolkit_data,
            self.snapshot_name)
        log.info("MessWorker %s - analysis started" % self.id)

        assert(self.mess.get_vm_state(self.vm_name) == "ANALYSING")
        for i in range(self.sleep_time_minutes):
            time.sleep(60)
            assert(self.mess.get_vm_state(self.vm_name) == "ANALYSING")

        # TODO: when should force_stop be True?
        log.info("MessWorker %s - stopping analysis" % self.id)
        force_stop = False
        result_file_name = self.mess.stop_analysis(self.vm_name, force_stop)
        log.info("MessWorker %s - analysis stopped" % self.id)
        if not force_stop:
            result_file_url = "%s/%s" % (self.results_url, result_file_name)
            result_file_path = os.path.join(self.output_base_dir, args.dir,
                                            result_file_name)
            directory = os.path.dirname(result_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            urllib.request.urlretrieve(result_file_url, result_file_path)
            log.info("MessWorker %s - saved sample to %s" %
                     (self.id, result_file_path))


class Sample(object):
    def __init__(self, path, entry_func, target_name):
        self.path = path
        self.entry_func = entry_func
        self.target_name = target_name

        if self.entry_func == "":
            self.command = "!mpath/%s" % self.target_name
        else:
            self.command = "rundll32.exe%%!mpath/%s,%s" % (self.target_name,
                                                          self.entry_func)

    def load(self):
        with open(self.path, "rb") as f:
            data = xmlrpc.client.Binary(f.read())
        return data


def start_workers(config, queue):
    toolkit = config.get("Analyze", "toolkit")
    output_base_dir = config.get("Analyze", "output_base_dir")
    sleep_time_minutes = int(config.get("Analyze", "sleep_time_minutes"))
    results_url = config.get("MESS", "results_url")
    proxy_url = config.get("MESS", "proxy_url")
    mess_client = xmlrpc.client.ServerProxy(proxy_url)

    workers = (config.get("Analyze", "workers")).split(",")
    for (i, worker) in enumerate(workers):
        vm_name = config.get(worker, "vm_name")
        snapshot_name = config.get(worker, "snapshot_name")
        m = MessWorker(i, queue, mess_client=mess_client, toolkit=toolkit,
                       output_base_dir=output_base_dir,
                       sleep_time_minutes=sleep_time_minutes,
                       results_url=results_url, vm_name=vm_name,
                       snapshot_name=snapshot_name)
        m.start()


class SampleDescriptor(object):

    def __init__(self):
        self.entryfunc_map = {}
        pass

    def parse(self, descriptor_file):
        current_func = ""
        with open(descriptor_file, "r") as f:
            for line in f:
                if "EntryFunction" in line:
                    current_func = (line.split(":")[-1]).strip()
                    if re.match('[^\w\W_ ]', current_func):
                        # match anything that isn't a valid function name
                        log.warn("Found invalid entry function name: %s. "
                                 "Ignoring... " % current_func)
                        current_func = ""
                elif ".exe" in line:
                    filename = next(filter(lambda word: ".exe" in word,
                                      (line.split(" ")))).strip()
                    if filename in self.entryfunc_map.keys():
                        log.warn("Duplicate entry function for sample %s" %
                                 filename)
                    self.entryfunc_map[filename] = current_func
                    if current_func != "":
                        log.debug("%s - entry function is %s" % (filename,
                                                                 current_func))
                    else:
                        log.debug("%s has no entry function" % filename)
                elif line in ['\n', '\r\n']:
                    current_func = ""


def init_logger(config):
    log_cfg_path = config.get("Common", "logging_cfg")
    logging.config.fileConfig(log_cfg_path)


def main():
    config = configparser.ConfigParser()
    config.read([args.config])

    init_logger(config)

    sample_target_name = config.get("Common", "sample_target_name")

    samples_queue = queue.Queue()
    start_workers(config, samples_queue)

    input_base_dir = config.get("Analyze", "input_base_dir")

    txt_glob = glob.glob("%s/%s/**/*.txt" % (input_base_dir, args.dir),
                         recursive=True)
    log.info("Discovered %s txt files" % len(txt_glob))
    if len(txt_glob) == 0:
        log.warn("Didn't discover any .txt files with possible descriptors!")

    sample_descriptor = SampleDescriptor()
    for descriptor_file in txt_glob:
        sample_descriptor.parse(descriptor_file)

    log.info("Discovered %s entry functions across all samples" %
              len(sample_descriptor.entryfunc_map.keys()))


    exe_glob = glob.glob("%s/%s/**/*.exe" % (input_base_dir, args.dir),
                         recursive=True)
    log.info("Discovered %s samples" % len(exe_glob))
    for sample_path in exe_glob:
        log.debug("Enqueued sample: %s" % sample_path)

        sample_filename = os.path.basename(sample_path)
        entry_func = sample_descriptor.entryfunc_map.get(sample_filename)

        s = Sample(sample_path, entry_func, sample_target_name)
        samples_queue.put(s)

    samples_queue.join()


if __name__ == "__main__":
    main()