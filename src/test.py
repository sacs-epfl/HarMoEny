# from pynvml import *

# # Initialize NVML
# nvmlInit()

# try:
#     # Get the handle for the first GPU (index 0)
#     device = nvmlDeviceGetHandleByIndex(0)

#     # Loop through all possible NVLink connections (up to 6 links per device)
#     for link in range(6):
#         try:
#             # Check if the link supports P2P traffic
#             is_p2p_supported = nvmlDeviceGetNvLinkCapability(device, link, 0)
#             print(f"Link {link}: P2P Supported: {bool(is_p2p_supported)}")
            
#             # Check if the link supports remote atomics
#             is_remote_atomic_supported = nvmlDeviceGetNvLinkCapability(device, link, 1)
#             print(f"Link {link}: Remote Atomics Supported: {bool(is_remote_atomic_supported)}")
            
#             # Check if the link supports system memory access
#             is_sysmem_supported = nvmlDeviceGetNvLinkCapability(device, link, 2)
#             print(f"Link {link}: System Memory Access Supported: {bool(is_sysmem_supported)}")
#         except NVMLError_NotSupported:
#             print(f"Link {link}: Capability query not supported.")
#         except NVMLError as error:
#             print(f"Error querying link {link}: {error}")

# finally:
#     # Shutdown NVML
#     nvmlShutdown()

from pynvml import *

nvmlInit()
device = nvmlDeviceGetHandleByIndex(0)

try:
    # Reset counters for link 0, counter 0 and 1
    nvmlDeviceResetNvLinkUtilizationCounter(device, 0, 0)
    nvmlDeviceResetNvLinkUtilizationCounter(device, 0, 1)

    # Perform some GPU-to-GPU communication here to generate traffic

    # Query the counters
    rx_counter, tx_counter = nvmlDeviceGetNvLinkUtilizationCounter(device, 0, 0)
    print(f"Receive Counter: {rx_counter}, Transmit Counter: {tx_counter}")
except NVMLError as error:
    print(f"Error occurred: {error}")
finally:
    nvmlShutdown()

