import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy



# Check GPU availability and enable memory growth
def configureHardware():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU: {gpu}")
            # Enable memory growth for each GPU
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
    else:
        print("No GPUs found, using CPU.")

    # Use mixed precision for faster training on GPUs (Optional)
    try:
        set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    except Exception as e:
        print(f"Mixed precision not supported: {e}")
