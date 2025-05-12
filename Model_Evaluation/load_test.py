import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Your Flask /predict URL
FLASK_URL = "http://localhost:8000/predict"   # or replace localhost with your host/IP

# Payloads per borough (we’ll round-robin through these)
boroughs = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island"
]

load_pattern = [1, 2, 3, 5, 3, 2, 1]  # concurrent workers per stage
delay_between_steps = 30              # seconds per stage

def send_continuous_requests(borough, duration_sec):
    """
    Keeps sending POST /predict for the given borough 
    until duration_sec has elapsed.
    """
    end = time.time() + duration_sec
    while time.time() < end:
        try:
            resp = requests.post(
                FLASK_URL,
                json={"borough": borough},
                timeout=5
            )
            # you can inspect resp.status_code or resp.json() here if you like
        except Exception:
            pass  # ignore errors for load testing

def run_load_stage(concurrent_workers, duration_sec, borough):
    """
    Spins up `concurrent_workers` threads, each sending 
    continuous requests for `duration_sec` seconds.
    """
    with ThreadPoolExecutor(max_workers=concurrent_workers) as ex:
        futures = [
            ex.submit(send_continuous_requests, borough, duration_sec)
            for _ in range(concurrent_workers)
        ]
        for f in futures:
            f.result()  # wait for each thread to finish

# Main loop: ramp up and down
for stage, workers in enumerate(load_pattern):
    # Pick a borough (round-robin)
    borough = boroughs[stage % len(boroughs)]
    print(f"[Stage {stage+1}/{len(load_pattern)}] "
          f"{workers} workers → {borough} for {delay_between_steps}s")
    run_load_stage(workers, delay_between_steps, borough)
    print("  stage complete\n")
