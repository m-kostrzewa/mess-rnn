from common import *

import csv
import shutil
import tempfile
import zipfile

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


class Worker(threading.Thread):
    def __init__(self, id, queue):
        super().__init__()
        self.id = id
        self.queue = queue
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
        log.info("Worker %s - inspecting %s" % (self.id, zip_sample.rel_path))

        with tempfile.SpooledTemporaryFile(max_size=1024*1024) as tmpfile:
            self.extract_pml(zip_sample, tmpfile)
            return

    def extract_pml(self, zip_sample, tmpfile):
        zip_abs_path = os.path.join(IN_BASE_DIR, zip_sample.rel_path)
        with zipfile.ZipFile(zip_abs_path) as zip_file:
            found_files = zip_file.namelist()
            log.debug("Worker %s - files found in the archive: %s" %
                        (self.id, found_files))

            # TODO: this should be a .csv. Use .pml for PoC
            pml_file = next(filter(lambda filepath: ".pml" == \
                                                os.path.splitext(filepath)[1],
                                   found_files))
            log.debug("Worker %s - pml file: %s" % (self.id, pml_file))

            with zip_file.open(pml_file) as pml:
                shutil.copyfileobj(pml, tmpfile)
                log.info("Worker %s - extracted %s" % (self.id, pml_file))


class ZippedSample(object):
    def __init__(self, rel_path, target_name):
        self.rel_path = rel_path
        self.target_name = target_name


def start_workers(config, queue):
    num_workers = config.get("Preprocess", "num_workers")
    worker_threads = []
    for i in num_workers:
        w = Worker(i, queue)
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

    zip_queue = queue.Queue()
    worker_threads = start_workers(config, zip_queue)

    sample_target_name = config.get("Common", "sample_target_name")
    enqueue_zips(zip_queue, sample_target_name)

    zip_queue.join()
    stop_workers(zip_queue, worker_threads)


if __name__ == "__main__":
    main()

# example line in .csv from previous step:
# "9:58:55.6960949 AM","CompatTelRunner.exe","2392","ReadFile"
# "C:\Windows\System32\appraiser.dll","SUCCESS",
# "Offset: 1,193,472, Length: 14,848, I/O Flags: Non-cached, Paging I/O, Synchronous # Paging I/O, Priority: Normal"
