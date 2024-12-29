import torch.multiprocessing as mp 
import os
import pandas as pd
import time
import logging
from threading import Thread
import pynvml as nvml
import psutil

logger = logging.getLogger(__name__)

class Stats:
    class Collector:
        def __init__(self, name, func):
            self.name = name
            self.func = func
            self.thread = None
            self.stop_event = mp.Event()
            self.stats = []
        
        def start(self):
            logger.info(f"Starting stat collection thread for {self.name}")
            thread = Thread(target=self.func, args=(self.stop_event, self.stats))
            thread.start()
            self.thread = thread
        
        def stop(self):
            logger.info(f"Stopping stat collection thread for {self.name}")
            self.stop_event.set()
            self.thread.join()

        def save(self, path):
            logger.info(f"Saving stat collection for {self.name}")
            df = pd.DataFrame(self.stats)
            df.to_csv(os.path.join(path, f"{self.name}.csv"))
            logger.info(f"Finished saving stat collection for {self.name}")
    
    def __init__(self, gpu=False, cpu=False, num_gpus=8, rate=1):
        self.gpu = gpu
        self.cpu = cpu
        self.num_gpus = num_gpus
        self.rate = rate

        self.stop_event = mp.Event()
        self.threads = []

        self.collectors = []
        if self.gpu:
            nvml.nvmlInit()
            self.collectors.append(self.Collector("gpu", self.collect_gpu_stats))
        if self.cpu:
            self.collectors.append(self.Collector("cpu", self.collect_cpu_stats))
    
    def start(self):
        for collector in self.collectors:
            collector.start()
    
    def stop(self):
        for collector in self.collectors:
            collector.stop()
        if self.gpu:
            nvml.nvmlShutdown()
    
    def save(self, path="."):
        for collector in self.collectors:
            collector.save(path)
    
    def collect_gpu_stats(self, stop_event, stats_list):
        handles = [nvml.nvmlDeviceGetHandleByIndex(index) for index in range(self.num_gpus)]

        while not stop_event.is_set():
            stats_list.append({
                "timestamp": time.time(),
                "gpu_util": [nvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in handles],
                "gpu_mem_used": [nvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles],
            })

            time.sleep(self.rate)
    
    def collect_cpu_stats(self, stop_event, stats_list):
        while not stop_event.is_set():
            stats_list.append({
                "timestamp": time.time(),
                "cpu_util": psutil.cpu_percent(interval=None),
                "cpu_mem_used": psutil.virtual_memory().used,
            })

            time.sleep(self.rate)