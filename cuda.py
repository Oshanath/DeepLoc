import torch

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Get the number of available CUDA devices
    cuda_device_count = torch.cuda.device_count()
    print(f"CUDA is available with {cuda_device_count} device(s).")

    # Print information about each CUDA device
    for i in range(cuda_device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")

else:
    print("CUDA is not available. Using CPU.")