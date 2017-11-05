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

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True,
                    help="Directory containing raw .csv files relative to "
                         "input base dir specified in config file. It is "
                         "also the name of output directory under base output "
                         "dir, also specified in config file.")
parser.add_argument("--config", type=str,
                    help="Filepath to config file.")
args = parser.parse_args()


IN_BASE_DIR = "."
OUT_BASE_DIR = "."


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
    def __init__(self, id, queue, op_dict):
        super().__init__()
        self.id = id
        self.queue = queue
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

        csv_file = self.extract_csv(zip_sample.rel_path)
        benevolent, malevolent = self.encode(csv_file, zip_sample.target_name)
        out_dir = os.path.join(OUT_BASE_DIR,
                               os.path.dirname(zip_sample.rel_path))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.save_encodings(benevolent, os.path.join(out_dir,
                                                     BENEVOLENT_FILENAME))
        self.save_encodings(malevolent, os.path.join(out_dir,
                                                     MALEVOLENT_FILENAME))

    def extract_csv(self, rel_path):
        zip_abs_path = os.path.join(IN_BASE_DIR, rel_path)
        with zipfile.ZipFile(zip_abs_path) as zip_file:
            found_files = zip_file.namelist()
            log.debug("Worker %s - files found in the archive: %s" %
                        (self.id, found_files))

            csv_file = next(filter(lambda filepath: ".csv" == \
                                   os.path.splitext(filepath)[1].lower(),
                                   found_files))
            log.debug("Worker %s - opening csv file: %s" % (self.id, csv_file))
            return zip_file.open(csv_file, mode="r")

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
        is_dll_malware = row["Process Name"] == "svchost.exe" and \
                         "rundll" in row["Path"] and \
                         target_name in row["Path"]
        is_exe_malware = row["Process Name"] == target_name
        return is_dll_malware or is_exe_malware

    def save_encodings(self, enc, out_path):
        log.debug("Worker %s - saving %s process encodings to %s" %
                  (self.id, len(enc.keys()), out_path))
        with open(out_path, "w") as f:
            for pid, encodings in enc.items():
                f.write("%s\n" % ",".join(map(str, encodings)))


class ZippedSample(object):
    def __init__(self, rel_path, target_name):
        self.rel_path = rel_path
        self.target_name = target_name


def start_workers(config, queue, op_dict):
    num_workers = config.get("Preprocess", "num_workers")
    worker_threads = []
    for i in range(int(num_workers)):
        w = Worker(i, queue, op_dict)
        w.start()
        worker_threads.append(w)
    return worker_threads


def enqueue_zips(queue, sample_target_name):
    zip_glob = glob.glob("%s/%s/**/*.zip" % (IN_BASE_DIR, args.dir),
                         recursive=True)

    log.info("Discovered %s zips" % len(zip_glob))
    for zip_path in zip_glob:
        if not zipfile.is_zipfile(zip_path):
            log.error("%s - magic number is invalid! Skippping." % zip_path)
        rel_path = zip_path.replace(IN_BASE_DIR, ".")
        zs = ZippedSample(rel_path, sample_target_name)
        queue.put(zs)
        log.debug("Enqueued zip: %s" % zs.rel_path)


def stop_workers(queue, threads):
    for i in range(len(threads)):
       queue.put(None)
    for t in threads:
        t.join()


def main():
    global IN_BASE_DIR, OUT_BASE_DIR

    config = configparser.ConfigParser()
    config.read([args.config])
    init_logger(config)
    IN_BASE_DIR = config.get("Preprocess", "input_base_dir")
    OUT_BASE_DIR = config.get("Preprocess", "output_base_dir")

    dict_path = config.get("Common", "dictionary_path")
    op_dict = OperationDict(dict_path)

    zip_queue = queue.Queue()
    worker_threads = start_workers(config, zip_queue, op_dict)

    sample_target_name = config.get("Common", "sample_target_name")
    enqueue_zips(zip_queue, sample_target_name)

    zip_queue.join()
    stop_workers(zip_queue, worker_threads)


if __name__ == "__main__":
    main()

