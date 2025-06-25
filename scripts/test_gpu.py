import tensorflow as tf


def main():
    """
    Checks for available GPUs using TensorFlow and prints the information.
    """
    print("Checking for GPUs...")
    print("=" * 50)
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}:")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details and "name" in details:  # Check if 'name' key exists
                    print(f"    Name: {details['name']}")
                else:
                    print(
                        f"    Name: Not available (PhysicalDevice: {gpu.name})"
                    )  # Fallback if name is not in details
                print(f"    Type: {gpu.device_type}")
            except Exception as e:
                print(f"    Could not get details for GPU {i}: {e}")
                print(f"    Name: {gpu.name} (PhysicalDevice)")
                print(f"    Type: {gpu.device_type}")

    else:
        print("No GPUs found by TensorFlow.")

    # Additional check for CUDA availability, often useful
    cuda_available = tf.test.is_built_with_cuda()
    print(f"\nTensorFlow built with CUDA: {cuda_available}")
    if cuda_available and gpus:
        print(
            "CUDA should be available for TensorFlow operations on the detected GPU(s)."
        )
    elif cuda_available and not gpus:
        print("TensorFlow is built with CUDA, but no GPUs were detected by TensorFlow.")
        print("Please check your CUDA installation and GPU drivers.")
    elif not cuda_available and gpus:
        print(
            "Warning: GPUs detected, but TensorFlow is not built with CUDA. GPU acceleration will not be available."
        )
    else:
        print("TensorFlow is not built with CUDA, and no GPUs were detected.")

    print("=" * 50)


if __name__ == "__main__":
    main()
