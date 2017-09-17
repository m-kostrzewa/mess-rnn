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

log = logging.getLogger("analyze")

parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", type=str, required=True,
                    help="Directory containing raw input samples.")
parser.add_argument("--out-dir", type=str, required=True,
                    help="Directory that shall contain output .zips.")
parser.add_argument("--config", type=str,
                    help="Filepath of .conf file.")
args = parser.parse_args()


class MessWorker(threading.Thread):
    def __init__(self, id, queue, mess_client, toolkit, sleep_time_minutes,
                 results_url, vm_name, snapshot_name):
        super().__init__()
        self.id = id
        self.queue = queue
        self.mess = mess_client
        self.toolkit = toolkit
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
            log.info("MessWorker %s - task done" % self.id)

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
            result_file_path = os.path.join(args.out_dir, result_file_name)
            urllib.request.urlretrieve(result_file_url, result_file_path)
            log.info("MessWorker %s - saved sample to %s" %
                     (self.id, result_file_path))


class Sample(object):
    def __init__(self, path, target_name):
        # TODO handle custom launch commands
        # mess-start-analysis.py -s 987t67g.exe -n locky.exe -t ./toolkits/basic.zip
        # -m MW2 -a MESS-SNAPSHOT-MW2 -c 'rundll32.exe%!mpath/locky.exe,aqua'
        self.path = path
        self.target_name = target_name
        self.command = "!mpath/!mname"
        self.command = self.command.replace("!mname", self.target_name)

    def load(self):
        with open(self.path, "rb") as f:
            data = xmlrpc.client.Binary(f.read())
        return data


def start_workers(config, queue):
    toolkit = config.get("Analyze", "toolkit")
    sleep_time_minutes = int(config.get("Analyze", "sleep_time_minutes"))
    results_url = config.get("MESS", "results_url")
    proxy_url = config.get("MESS", "proxy_url")
    mess_client = xmlrpc.client.ServerProxy(proxy_url)

    workers = (config.get("Analyze", "workers")).split(",")
    for (i, worker) in enumerate(workers):
        vm_name = config.get(worker, "vm_name")
        snapshot_name = config.get(worker, "snapshot_name")
        m = MessWorker(i, queue, mess_client=mess_client, toolkit=toolkit,
                       sleep_time_minutes=sleep_time_minutes,
                       results_url=results_url, vm_name=vm_name,
                       snapshot_name=snapshot_name)
        m.start()


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

    exe_glob = glob.glob("%s/**/*.exe" % args.in_dir, recursive=True)
    log.info("Discovered %s samples" % len(exe_glob))
    for sample in exe_glob:
        log.debug("Enqueued sample: %s" % sample)
        s = Sample(sample, sample_target_name)
        samples_queue.put(s)

    samples_queue.join()


if __name__ == "__main__":
    main()