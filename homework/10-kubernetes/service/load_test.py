import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "http://localhost:30080/predict"
PAYLOAD = {"url": "http://bit.ly/mlbookcamp-pants"}

WORKERS = 20            # requests sent at the same time
DURATION_SECONDS = 180  # how long to keep sending

ok = 0
failed = 0


def send_requests(worker_id):
    global ok, failed
    end = time.time() + DURATION_SECONDS
    while time.time() < end:
        try:
            r = requests.post(URL, json=PAYLOAD, timeout=60)
            if r.status_code == 200:
                ok += 1
            else:
                failed += 1
        except Exception:
            failed += 1


print(f"Sending load for {DURATION_SECONDS} seconds with {WORKERS} workers...")
start = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for i in range(WORKERS):
        pool.submit(send_requests, i)

elapsed = time.time() - start
print(f"Done in {elapsed:.0f}s: {ok} successful, {failed} failed requests")