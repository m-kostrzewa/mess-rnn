#!/usr/bin/python3
#
# Unpacks zipped results from MESS analysis. Doing so, updates the dictionary
# of encountered function calls and saves their encoded sequence for each
# PID, dividing them into malevolent and benevolent.
#

from common import *

import csv
import shutil
import zipfile
import io
from collections import defaultdict

log = logging.getLogger("preprocess")
config = configparser.ConfigParser()


def main():
    args = parse_args()
    init(args.config)
    in_subdir = args.input

    in_base_dir = config.get("Workspace", "analyzed_base_dir")
    in_abs_dir = os.path.join(in_base_dir, in_subdir)

    zips_paths = find_files_recursive(in_abs_dir, "zip")
    target_name = config.get("Common", "sample_target_name")

    preprocess(zips_paths, target_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to directory containing zipfiles (output "
                             "of MESS system) with .csv Process Monitor files. "
                             "Must be relative to analyzed_base_dir in config.")
    parser.add_argument("--config", type=str, default="mess-rnn.cfg",
                        help="Config filepath.")
    return parser.parse_args()


def init(config_path):
    config.read([config_path])
    init_logger(config)


def preprocess(zip_paths, target_name):
    dict_path = config.get("Workspace", "dictionary_path")
    op_dict = OperationDict(dict_path)

    zip_base_dir = config.get("Workspace", "analyzed_base_dir")
    out_base_dir = config.get("Workspace", "encoded_base_dir")

    zip_queue = queue.Queue()
    output_queue = queue.Queue()
    worker_threads = start_workers(config, zip_queue, output_queue,
                                   zip_base_dir, out_base_dir, op_dict)

    zips = make_zips(zip_paths, zip_base_dir, target_name)
    enqueue_zips(zip_queue, zips)

    zip_queue.join()
    stop_workers(zip_queue, worker_threads)

    out_paths = read_all_queue(output_queue)
    log.info("Output encoding files: %s" % out_paths)
    if len(out_paths) < 2:
        log.warning("Expected at least two output encoding files!")
    return out_paths


def start_workers(config, queue, output_queue, in_base_dir, out_base_dir, 
                 op_dict):
    num_workers = config.get("Preprocess", "num_workers")
    worker_threads = []
    for i in range(int(num_workers)):
        w = Worker(i, queue, output_queue, in_base_dir, out_base_dir, op_dict)
        w.start()
        worker_threads.append(w)
    return worker_threads


def make_zips(zip_paths, zip_base_dir, target_name):
    zips = []
    for zip_path in zip_paths:
        log.error(zip_path)
        if not zipfile.is_zipfile(zip_path):
            log.error("%s - magic number is invalid! Skippping." % zip_path)
        rel_path = zip_path.replace(zip_base_dir, "./")
        zs = ZippedSample(rel_path, target_name)
        zips.append(zs)
    log.info("Discovered %s zips" % len(zips))
    return zips


def enqueue_zips(queue, zips):
    for zs in zips:
        queue.put(zs)
        log.debug("Enqueued zip: %s" % zs.rel_path)


def stop_workers(queue, threads):
    for i in range(len(threads)):
        queue.put(None)
    for t in threads:
        t.join()


class OperationDict(object):
    def __init__(self, dict_path):
        # TODO: persistence
        self.lock = threading.Lock()
        self.dict_path = dict_path
        self.dict = load_operation_dict(dict_path)

    def get(self, key):
        with self.lock:
            if key not in self.dict.keys():
                new_idx = self.find_first_unused_idx()
                self.dict[key] = new_idx
                with open(self.dict_path, "a+") as persistent_dict:
                    persistent_dict.write("%s:%s\n" % (key, new_idx))
                log.debug("'%s' not in dict. Saved it. Index = %s" %
                          (key, new_idx))
            return self.dict[key]

    def find_first_unused_idx(self):
        used_idx = self.dict.values()
        if not self.dict.values(): return 0
        i = 0
        for i in range(max(used_idx)):
            if i not in used_idx:
                return i
        return max(used_idx) + 1


class Worker(threading.Thread):
    __lock = threading.Lock()

    def __init__(self, id, queue, output_queue, in_base_dir, out_base_dir, 
                 op_dict):
        super().__init__()
        self.id = id
        self.queue = queue
        self.output_queue = output_queue
        self.in_base_dir = in_base_dir
        self.out_base_dir = out_base_dir
        self.op_dict = op_dict
        log.info("Starting Worker thread %s" % self.id)

    def run(self):
        while True:
            zip_sample = self.queue.get()
            if zip_sample is None:
                log.warn("Worker %s - empty item, done" % self.id)
                break
            self.preprocess(zip_sample)
            log.info("Worker %s - job completed" % self.id)
            self.queue.task_done()

    def preprocess(self, zip_sample):
        log.info("Worker %s - processing %s" % (self.id, zip_sample.rel_path))

        out_dir = os.path.join(self.out_base_dir,
                               os.path.dirname(zip_sample.rel_path))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        csv_files = self.csv_extractor(zip_sample.rel_path)
        for csv_file in csv_files:
            try:
                benevolent, malevolent = self.encode(csv_file,
                                                     zip_sample.target_name)
                self.save_encodings(benevolent, os.path.join(out_dir,
                                                             OTHERS_FILENAME))
                self.save_encodings(malevolent, os.path.join(out_dir,
                                                             TARGET_FILENAME))
            finally:
                csv_file.close()

    def csv_extractor(self, rel_path):
        zip_abs_path = os.path.join(self.in_base_dir, rel_path)
        with zipfile.ZipFile(zip_abs_path) as zip_file:
            found_files = zip_file.namelist()
            log.debug("Worker %s - files found in the archive: %s" %
                        (self.id, found_files))

            found_csvs = filter(lambda filepath: ".csv" == \
                                    os.path.splitext(filepath)[1].lower(),
                                found_files)
            for csv in found_csvs:
                log.debug("Worker %s - opening csv file: %s" % (self.id, csv))
                yield zip_file.open(csv, mode="r")

    def encode(self, csv_file, target_name):
        log.debug("Worker %s - encoding" % self.id)
        benevolent_encodings = defaultdict(list)
        malevolent_encodings = defaultdict(list)
        items_file  = io.TextIOWrapper(csv_file, encoding="iso-8859-1",
                                       newline="\r\n")
        for i, row in enumerate(csv.DictReader(items_file)):
            if i == 0: continue # skip column names
            enc_op = self.op_dict.get(row["Operation"])

            is_malware = self.is_malware(row, target_name)

            pid = row["PID"]
            if is_malware:
                malevolent_encodings[pid].append(enc_op)
            else:
                benevolent_encodings[pid].append(enc_op)
        return (benevolent_encodings, malevolent_encodings)

    def is_malware(self, row, target_name):
        target_name_in_path = target_name in row["Path"]
        target_is_process = row["Process Name"] == target_name
        return target_name_in_path or target_is_process

    def save_encodings(self, enc, out_path):
        with Worker.__lock:
            log.debug("Worker %s - saving %s process encodings to %s" %
                    (self.id, len(enc.keys()), out_path))
            self.output_queue.put(out_path)
            with open(out_path, "a") as f:
                for pid, encodings in enc.items():
                    f.write("%s\n" % ",".join(map(str, encodings)))


class ZippedSample(object):
    def __init__(self, rel_path, target_name):
        self.rel_path = rel_path
        self.target_name = target_name


if __name__ == "__main__":
    main()
