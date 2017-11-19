#!/usr/bin/python3
#
# Schedules analysis of executable sample files in MESS sandbox and
# collects results.
#

from common import *

import xmlrpc.client
import os
import urllib.request
import re
import threading

log = logging.getLogger("analyze")
config = configparser.ConfigParser()


def main():
    args = parse_args()
    init(args.config)
    in_subdir = args.input

    in_base_dir = config.get("Workspace", "raw_base_dir")
    in_abs_dir = os.path.join(in_base_dir, in_subdir)

    samples_paths = find_files_recursive(in_abs_dir, "exe")
    descriptors_paths = find_files_recursive(in_abs_dir, "txt")

    analyze(samples_paths, descriptors_paths)


def init(config_path):
    config.read([config_path])
    init_logger(config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input sample or directory containing "
                             "samples. Relative to raw_base_dir in config.")
    parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                        help="Config filepath.")
    return parser.parse_args()


def find_files_recursive(directory, filetype):
    return glob.glob("%s/*.%s" % (directory, filetype), recursive=True)


def analyze(samples_paths, descriptors_paths):
    descriptor = load_sample_descriptors(descriptors_paths)

    target_name = config.get("Common", "sample_target_name")
    samples_base_dir = config.get("Workspace", "raw_base_dir")
    samples = make_samples(samples_paths, samples_base_dir, target_name,
                           descriptor)

    samples_queue = queue.Queue()
    out_base_dir = config.get("Workspace", "analyzed_base_dir")
    worker_threads = start_workers(config, samples_queue, out_base_dir)

    enqueue_samples(samples_queue, samples)
    samples_queue.join()

    stop_workers(samples_queue, worker_threads)


def load_sample_descriptors(descriptors_paths):
    log.info("Discovered %s txt files" % len(descriptors_paths))
    if len(descriptors_paths) == 0:
        log.warn("Didn't discover any .txt files with possible descriptors!")

    sample_descriptor = SampleDescriptor()
    for descriptor_file in descriptors_paths:
        sample_descriptor.parse(descriptor_file)

    log.info("Discovered %s entry functions across all samples" %
              len(set(sample_descriptor.entryfunc_map.values())))
    return sample_descriptor


def make_samples(samples_paths, samples_base_dir, target_name, descriptor):
    samples = []
    for sample_path in samples_paths:
        rel_path = sample_path.replace(samples_base_dir, "./")

        sample_filename = os.path.basename(rel_path)
        entry_func = descriptor.entryfunc_map.get(sample_filename)

        s = Sample(samples_base_dir, rel_path, entry_func, target_name)
        samples.append(s)
    log.info("Discovered %s samples" % len(samples))
    return samples


def start_workers(config, queue, out_base_dir):
    toolkit = config.get("Analyze", "toolkit")
    sleep_time_minutes = int(config.get("Analyze", "sleep_time_minutes"))
    results_url = config.get("MESS", "results_url")
    proxy_url = config.get("MESS", "proxy_url")

    workers = (config.get("Analyze", "workers")).split(",")
    worker_threads = []
    for (i, worker) in enumerate(workers):
        vm_name = config.get(worker, "vm_name")
        snapshot_name = config.get(worker, "snapshot_name")
        m = MessWorker(i, queue, proxy_url=proxy_url, toolkit=toolkit,
                       sleep_time_minutes=sleep_time_minutes,
                       results_url=results_url, vm_name=vm_name,
                       snapshot_name=snapshot_name, out_base_dir=out_base_dir)
        m.start()
        worker_threads.append(m)
    return worker_threads


def enqueue_samples(samples_queue, samples):
    for sample in samples:
        samples_queue.put(sample)
        log.debug("Enqueued sample: %s" % sample.rel_path)


def stop_workers(queue, threads):
    for i in range(len(threads)):
        queue.put(None)
    for t in threads:
        t.join()


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


class Sample(object):
    def __init__(self, base_dir, rel_path, entry_func, target_name):
        self.rel_path = rel_path
        self.entry_func = entry_func
        self.target_name = target_name
        self.abs_path = os.path.join(base_dir, self.rel_path)

        if self.entry_func == "":
            self.command = "!mpath/%s" % self.target_name
        else:
            self.command = "rundll32.exe%%!mpath/%s,%s" % (self.target_name,
                                                           self.entry_func)

    def load(self):
        with open(self.abs_path, "rb") as f:
            data = xmlrpc.client.Binary(f.read())
        return data


class MessWorker(threading.Thread):
    def __init__(self, id, queue, proxy_url, toolkit, sleep_time_minutes, 
                 results_url, vm_name, snapshot_name, out_base_dir):
        super().__init__()
        self.id = id
        self.queue = queue
        self.proxy_url = proxy_url
        self.mess = xmlrpc.client.ServerProxy(proxy_url)
        self.toolkit = toolkit
        self.sleep_time_minutes = sleep_time_minutes
        self.results_url = results_url
        self.vm_name = vm_name
        self.snapshot_name = snapshot_name
        self.out_base_dir = out_base_dir
        log.info("Starting MessWorker thread %s" % self.id)
        log.debug("MessWorker %s - vm_name=%s, snapshot_name=%s" %
                  (self.id, self.vm_name, self.snapshot_name))

    def run(self):
        while True:
            sample = self.queue.get()
            if sample is None:
                log.warn("MessWorker %s - empty sample, done" % self.id)
                break
            self.analyze(sample)
            log.info("MessWorker %s - job completed" % self.id)
            self.queue.task_done()

    def analyze(self, sample):
        if self.mess.get_vm_state(self.vm_name) != "READY":
            log.warn("MessWorker %s - VM in Analysing state, will try to stop"
                      % self.id)
            force_stop = True
            self.mess.stop_analysis(self.vm_name, force_stop)
            if self.mess.get_vm_state(self.vm_name) != "READY":
                raise RuntimeError("Couldn't stop MESS VM")
            else:
                log.info("MessWorker %s - managed to stop VM. Continuing "
                         "scheduled analysis." % self.id)

        log.info("MessWorker %s - starting analysis of %s."
                 % (self.id, sample.rel_path))
        with open(self.toolkit, "rb") as f:
            toolkit_data = xmlrpc.client.Binary(f.read())

        self.mess.start_analysis(self.vm_name, sample.target_name,
            sample.command.split("%"), sample.load(), toolkit_data,
            self.snapshot_name)
        log.info("MessWorker %s - analysis started" % self.id)

        for i in range(self.sleep_time_minutes):
            if self.mess.get_vm_state(self.vm_name) != "ANALYSING":
                raise RuntimeError("MESS VM stopped analysis unexpectedly")
            time.sleep(60)

        # TODO: when should force_stop be True?
        log.info("MessWorker %s - stopping analysis" % self.id)
        force_stop = False
        result_filename = self.mess.stop_analysis(self.vm_name, force_stop)
        log.info("MessWorker %s - analysis stopped" % self.id)
        if not force_stop:
            out_dir = os.path.join(self.out_base_dir,
                                   os.path.dirname(sample.rel_path))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, result_filename)

            result_url = "%s/%s" % (self.results_url, result_filename)
            urllib.request.urlretrieve(result_url, out_path)
            log.info("MessWorker %s - saved sample to %s" % (self.id, out_path))


if __name__ == "__main__":
    main()
