import concurrent.futures
import math
import multiprocessing
import traceback
from pathlib import Path
from typing import *

from tqdm import tqdm


def process_tasks(task_fn: Callable, tasks: List[Any], num_workers: int, use_indexes: bool = False, **kwargs) -> Dict:
    print("Starting parallel execution with {} workers!".format(num_workers))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(task_fn, task, **kwargs): idx if use_indexes else task for idx, task in enumerate(tasks)}
        results = {}

        print("Awaiting completion!".format(num_workers))

        try:
            with tqdm(desc="Processing ...", total=len(futures), smoothing=0) as progress_bar:
                for f in concurrent.futures.as_completed(futures.keys()):
                    try:
                        results[futures[f]] = f.result()
                    except Exception:
                        print("EXCEPTION: ", traceback.format_exc())
                        print(f"FAILED TASK: {futures[f]}")
                    progress_bar.update(1)
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            exit(-1)

    return results

def get_cpus():
    try:
        cfs_quota_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        cfs_period_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return int(math.ceil(cfs_quota_us / cfs_period_us))
    except:
        pass
    return multiprocessing.cpu_count()
