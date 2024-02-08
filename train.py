import os
import boto3
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys
import threading
import re

model_name = os.getenv(
    "MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")
instance_dir = os.getenv("INSTANCE_DIR", "/images")
output_dir = os.getenv("OUTPUT_DIR", "/output/timber")
vae_path = os.getenv("VAE_PATH", "madebyollin/sdxl-vae-fp16-fix")
prompt = os.getenv("PROMPT", "timber boy")

bucket_name = os.getenv('BUCKET_NAME', None)
bucket_prefix = os.getenv('BUCKET_PREFIX', None)

s3 = boto3.client('s3')

os.makedirs(instance_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def unzip_to_sibling_folder(zip_file):
    # Get the name of the zip file without extension
    zip_file_name = os.path.splitext(zip_file)[0]

    # Create a folder with the same name as the zip file
    output_folder = os.path.join(os.path.dirname(zip_file), zip_file_name)
    os.makedirs(output_folder, exist_ok=True)

    # Construct the unzip command
    unzip_command = ['unzip', '-o', zip_file, '-d', output_folder]

    # Execute the unzip command
    try:
        subprocess.run(unzip_command, check=True)
        print(
            f"Zip file '{zip_file}' successfully extracted to '{output_folder}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to extract zip file '{zip_file}': {e}")


def load_existing_progress():
    if bucket_name is None or bucket_prefix is None:
        return

    try:
        # List objects in the bucket with the specified prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=bucket_prefix)

        # Filter objects to include only those with .zip extension
        zip_files = [obj for obj in response.get(
            'Contents', []) if obj['Key'].endswith('.zip')]

        # Sort the zip files by LastModified attribute in descending order
        sorted_zip_files = sorted(
            zip_files, key=lambda x: x['LastModified'], reverse=True)

        # Check if there are any zip files
        if sorted_zip_files:
            # Get the most recent zip file
            most_recent_zip = sorted_zip_files[0]
            most_recent_zip_key = most_recent_zip['Key']

            # Download the zip file to output directory, using the same name
            output_file = f"{output_dir}/{most_recent_zip_key.split('/')[-1]}"
            s3.download_file(bucket_name, most_recent_zip_key, output_file)

            # Unzip the downloaded file
            unzip_to_sibling_folder(output_file)

            # remove the zip file
            os.remove(output_file)
        else:
            print("No existing progress found.")

    except Exception as e:
        print(e)
        exit(1)


def train():
    command_array = [
        "accelerate", "launch", "train_dreambooth_lora_sdxl.py",
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={instance_dir}",
        f"--pretrained_vae_model_name_or_path={vae_path}",
        f"--output_dir={output_dir}",
        f"--instance_prompt=\"{prompt}\"",
        "--mixed_precision=fp16",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=500",
        "--checkpointing_steps=50",
        "--resume_from_checkpoint=latest",
        "--checkpoints_total_limit=1"
    ]

    print(f"Training command: {' '.join(command_array)}", flush=True)

    try:
        subprocess.run(command_array, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to train model: {e}")
        exit(1)


class MyHandler(FileSystemEventHandler):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_created(self, event):
        if event.is_directory and event.src_path.startswith(self.checkpoint_dir) and re.search(r"checkpoint-\d+/?$", event.src_path):
            print(
                f"New checkpoint directory created: {event.src_path}", flush=True)
            self.wait_for_write_completion(event.src_path)
            zip_and_upload_checkpoint(event.src_path)

    def wait_for_write_completion(self, directory):
        print(f"Waiting for write operations to stop in {directory}...")
        while True:
            time.sleep(1)
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        if entry.is_file() and entry.stat().st_size == 0:
                            # File size is 0, indicating ongoing write operation
                            break
                    else:
                        # No ongoing write operations found
                        print("Write operations stopped.")
                        return
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)


keep_alive = True


def monitor_checkpoint_directories(directory):
    global keep_alive
    event_handler = MyHandler(directory)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    print(
        f"Monitoring directory: {directory} for new 'checkpoint-*' subdirectories...")
    try:
        while keep_alive:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def zip_and_upload_checkpoint(checkpoint_dir):
    try:

        # Construct the zip file name, using the name of the rightmost directory
        # Get the name of the rightmost directory
        base_dir = os.path.basename(checkpoint_dir)

        # Construct the zip file name
        zip_file_name = f"{base_dir}.zip"

        print(
            f"Zipping and uploading checkpoint directory: {checkpoint_dir} as {zip_file_name}", flush=True)

        # Zip the contents of the rightmost directory
        zip_command = ['zip', '-r', zip_file_name, '-j', f"{checkpoint_dir}/*"]
        subprocess.run(zip_command, check=True, shell=True)

        # Upload the zip file to S3
        s3.upload_file(zip_file_name, bucket_name,
                       f"{bucket_prefix}{zip_file_name}")

        # Remove the zip file
        os.remove(zip_file_name)
        print(
            f"Checkpoint directory '{checkpoint_dir}' zipped and uploaded.", flush=True)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def upload_final_lora():
    try:
        # the file is $output_dir/pytorch_lora_weights.safetensors
        s3.upload_file(f"{output_dir}/pytorch_lora_weights.safetensors",
                       bucket_name, f"{bucket_prefix}/pytorch_lora_weights.safetensors")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Checking for existing progress...", flush=True)
    load_existing_progress()

    # Start training in one thread, and monitoring in another
    print("Starting training and monitoring threads...", flush=True)
    train_thread = threading.Thread(target=train)
    monitor_thread = threading.Thread(
        target=monitor_checkpoint_directories, args=(output_dir,))

    # wait for training to finish
    train_thread.start()
    monitor_thread.start()

    train_thread.join()
    keep_alive = False
    monitor_thread.join()

    print("Training and monitoring threads finished. Uploading Lora", flush=True)
    upload_final_lora()
