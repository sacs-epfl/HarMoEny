import sys
import time
import nvidia_smi
import signal

def collect_nvidia_stats(output_file):
        nvidia_smi.nvmlInit()
        stats = []
        running = True

        def signal_handler(signum, frame):
            nonlocal running
            print("Received termination signal. Saving data and shutting down....")
            running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while running:
                for i in range(NUM_GPUS):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                    utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    stats.append({
                        'timestamp': time.time(),
                        "gpu": i,
                        'gpu_utilization': utilization.gpu,
                        'memory_utilization': utilization.memory,
                        'memory_used': memory.used,
                        'memory_total': memory.total
                    })
                time.sleep(0.1) # Sleep is in seconds
        finally:
            nvidia_smi.nvmlShutdown()
            
            # Save stats to file
            with open(output_file, 'w') as f:
                f.write("timestamp,gpu,gpu_utilization,memory_utilization,memory_used,memory_total\n")
                for stat in stats:
                    f.write(f"{stat['timestamp']},{stat['gpu']},{stat['gpu_utilization']},{stat['memory_utilization']},{stat['memory_used']},{stat['memory_total']}\n")

            print(f"Data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 collect_nvidia_stats.py output_file")
        exit(1)

    collect_nvidia_stats(sys.argv[1])