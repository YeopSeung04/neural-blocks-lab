#!/usr/bin/env python3
import os
import signal
from pathlib import Path

from job_queue import JobManager
from job_tasks import JobTaskRunner


ROOT = Path(__file__).resolve().parent
DATABASE_TARGET = os.environ.get(
    "NBL_DATABASE_URL",
    os.environ.get("NBL_DATABASE_PATH", ROOT / ".data" / "neural_blocks.db"),
)
BASE_URL = os.environ.get("NBL_BASE_URL", "http://127.0.0.1:8770").rstrip("/")


def main():
    runner = JobTaskRunner(
        DATABASE_TARGET,
        base_url=BASE_URL,
        root=ROOT,
    )
    jobs = JobManager(
        DATABASE_TARGET,
        mode="redis",
        executor=runner,
    )
    recovered = jobs.recover_interrupted()
    print(f"Neural Blocks worker ready; recovered={recovered}", flush=True)
    running = True

    def stop_worker(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    while running:
        job = jobs.work_once(timeout=2)
        if job:
            print(
                f"job={job['id']} type={job['jobType']} status={job['status']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
