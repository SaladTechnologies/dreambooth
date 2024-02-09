import os
import boto3
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys
import threading
import re
import logging
import requests

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=log_level, format=log_format,
                    datefmt="%m/%d/%Y %H:%M:%S")

# Huggingface Hub Model Name or Path
model_name = os.getenv(
    "MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")

# Directory where training data is stored
instance_dir = os.getenv("INSTANCE_DIR", "/images")

# Directory where training output is stored
output_dir = os.getenv("OUTPUT_DIR", "/output")

# VAE model name or path
vae_path = os.getenv("VAE_PATH", "madebyollin/sdxl-vae-fp16-fix")
prompt = os.getenv("PROMPT", "photo of timberdog")

# Dreambooth training script from diffusers/examples/dreambooth
dreambooth_script = os.getenv(
    "DREAMBOOTH_SCRIPT", "train_dreambooth_lora_sdxl.py")

# Resolution of the images
resolution = os.getenv("RESOLUTION", "1024")

# Total number of training steps
max_train_steps = os.getenv("MAX_TRAIN_STEPS", "500")

# Save a checkpoint after every N steps
checkpointing_steps = os.getenv("CHECKPOINTING_STEPS", "50")

# S3 bucket and prefix for storing checkpoints
checkpoint_bucket_name = os.getenv('CHECKPOINT_BUCKET_NAME', None)
checkpoint_bucket_prefix = os.getenv('CHECKPOINT_BUCKET_PREFIX', None)

# S3 bucket and prefix for storing training data
data_bucket_name = os.getenv('DATA_BUCKET_NAME', None)
data_bucket_prefix = os.getenv('DATA_BUCKET_PREFIX', None)

# Webhook URLs and authentication headers
webhook_url = os.getenv("WEBHOOK_URL", None)
progress_webhook_url = os.getenv("PROGRESS_WEBHOOK_URL", webhook_url)
complete_webhook_url = os.getenv("COMPLETE_WEBHOOK_URL", webhook_url)

webhook_auth_header = os.getenv("WEBHOOK_AUTH_HEADER", None)
progress_webhook_auth_header = os.getenv(
    "PROGRESS_WEBHOOK_AUTH_HEADER", webhook_auth_header)
complete_webhook_auth_header = os.getenv(
    "COMPLETE_WEBHOOK_AUTH_HEADER", webhook_auth_header)

webhook_auth_value = os.getenv("WEBHOOK_AUTH_VALUE", None)
progress_webhook_auth_value = os.getenv(
    "PROGRESS_WEBHOOK_AUTH_VALUE", webhook_auth_value)
complete_webhook_auth_value = os.getenv(
    "COMPLETE_WEBHOOK_AUTH_VALUE", webhook_auth_value)

# Salad Machine and Container Group IDs
salad_machine_id = os.getenv("SALAD_MACHINE_ID", None)
salad_container_group_id = os.getenv("SALAD_CONTAINER_GROUP_ID", None)

s3 = boto3.client('s3')

os.makedirs(instance_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def download_data():
    if data_bucket_name is None or data_bucket_prefix is None:
        return

    try:
        # List objects in the bucket with the specified prefix
        response = s3.list_objects_v2(
            Bucket=data_bucket_name, Prefix=data_bucket_prefix)

        # Download the objects to the instance directory
        for obj in response.get('Contents', []):
            key = obj['Key']
            output_file = f"{instance_dir}/{key.split('/')[-1]}"
            s3.download_file(data_bucket_name, key, output_file)
            logging.info(f"Downloaded '{key}' to '{output_file}'.")

            if key.endswith('.zip'):
                # Unzip downloaded file into parent directory
                unzip_to_parent_folder(output_file)
                os.remove(output_file)
    except Exception as e:
        logging.error(e)
        exit(1)


def unzip_to_parent_folder(zip_file):
    # Construct the unzip command
    unzip_command = ['unzip', '-o', zip_file, '-d', os.path.dirname(zip_file)]

    # Execute the unzip command
    try:
        subprocess.run(unzip_command, check=True)
        logging.info(
            f"Zip file '{zip_file}' successfully extracted to '{os.path.dirname(zip_file)}'.")
    except subprocess.CalledProcessError as e:
        logging.info(f"Error: Failed to extract zip file '{zip_file}': {e}")


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
        logging.info(
            f"Zip file '{zip_file}' successfully extracted to '{output_folder}'.")
    except subprocess.CalledProcessError as e:
        logging.info(f"Error: Failed to extract zip file '{zip_file}': {e}")


def load_existing_progress():
    if checkpoint_bucket_name is None or checkpoint_bucket_prefix is None:
        return

    try:
        # List objects in the bucket with the specified prefix
        response = s3.list_objects_v2(
            Bucket=checkpoint_bucket_name, Prefix=checkpoint_bucket_prefix)

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
            s3.download_file(checkpoint_bucket_name,
                             most_recent_zip_key, output_file)

            # Unzip the downloaded file
            unzip_to_sibling_folder(output_file)

            # remove the zip file
            os.remove(output_file)
        else:
            logging.info("No existing progress found.")

    except Exception as e:
        logging.error(e)
        exit(1)


def train():
    command_array = [
        "accelerate", "launch", dreambooth_script,
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={instance_dir}",
        f"--pretrained_vae_model_name_or_path={vae_path}",
        f"--output_dir={output_dir}",
        f"--instance_prompt=\"{prompt}\"",
        "--mixed_precision=fp16",
        f"--resolution={resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={checkpointing_steps}",
        "--resume_from_checkpoint=latest",
        "--checkpoints_total_limit=1"
    ]

    logging.info(f"Training command: {' '.join(command_array)}")

    try:
        subprocess.run(command_array, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: Failed to train model: {e}")
        exit(1)


class MyHandler(FileSystemEventHandler):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_created(self, event):
        if event.is_directory and event.src_path.startswith(self.checkpoint_dir) and re.search(r"checkpoint-\d+/?$", event.src_path):
            logging.info(
                f"New checkpoint directory created: {event.src_path}")
            self.wait_for_write_completion(event.src_path)
            zip_and_upload_checkpoint(event.src_path)

    def wait_for_write_completion(self, directory):
        logging.info(f"Waiting for write operations to stop in {directory}...")
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
                        logging.info("Write operations stopped.")
                        return
            except Exception as e:
                logging.error(f"Error: {e}")
                sys.exit(1)


keep_alive = True


def monitor_checkpoint_directories(directory):
    global keep_alive
    event_handler = MyHandler(directory)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    logging.info(
        f"Monitoring directory: {directory} for new 'checkpoint-*' subdirectories...")
    try:
        while keep_alive:
            time.sleep(1)
        observer.stop()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def zip_and_upload_checkpoint(checkpoint_dir):
    try:
        # Get the name of the rightmost directory
        base_dir = os.path.basename(checkpoint_dir)

        # Construct the zip file name
        zip_file_name = f"{base_dir}.zip"

        logging.info(
            f"Zipping and uploading checkpoint directory: {checkpoint_dir} as {zip_file_name}")

        # Zip the contents of the rightmost directory
        zip_command = ['zip', '-rj',
                       f"{output_dir}/{zip_file_name}", f"{checkpoint_dir}"]
        logging.info(f"Running command: {' '.join(zip_command)}")
        subprocess.run(zip_command, check=True)

        # Upload the zip file to S3
        s3.upload_file(f"{output_dir}/{zip_file_name}", checkpoint_bucket_name,
                       f"{checkpoint_bucket_prefix}{zip_file_name}")
        send_progress_webhook(checkpoint_bucket_name,
                              f"{checkpoint_bucket_prefix}{zip_file_name}")

        # Remove the zip file
        os.remove(f"{output_dir}/{zip_file_name}")
        logging.info(
            f"Checkpoint directory '{checkpoint_dir}' zipped and uploaded.")
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


def upload_final_lora():
    try:
        # the file is $output_dir/pytorch_lora_weights.safetensors
        s3.upload_file(f"{output_dir}/pytorch_lora_weights.safetensors",
                       checkpoint_bucket_name, f"{checkpoint_bucket_prefix}pytorch_lora_weights.safetensors")
        send_complete_webhook(checkpoint_bucket_name,
                              f"{checkpoint_bucket_prefix}pytorch_lora_weights.safetensors")

    except Exception as e:
        logging.exception(f"Error: {e}")
        logging.error()
        sys.exit(1)


def send_webhook(url, auth_header, auth_value, bucket_name, key):
    try:
        if progress_webhook_url is None:
            return

        response = requests.post(url, json={
            "bucket_name": bucket_name,
            "key": key,
            "machine_id": salad_machine_id,
            "container_group_id": salad_container_group_id
        }, headers={
            auth_header: auth_value
        })
        response.raise_for_status()
        logging.info("Progress webhook sent successfully.")
    except Exception as e:
        logging.error(f"Error: {e}")


def send_progress_webhook(bucket_name, key):
    send_webhook(progress_webhook_url, progress_webhook_auth_header,
                 progress_webhook_auth_value, bucket_name, key)


def send_complete_webhook(bucket_name, key):
    send_webhook(complete_webhook_url, complete_webhook_auth_header,
                 complete_webhook_auth_value, bucket_name, key)


if __name__ == "__main__":
    logging.info("Checking for existing progress...")
    load_existing_progress()

    logging.info("Downloading training data...")
    download_data()

    # Start training in one thread, and monitoring in another
    logging.info("Starting training and monitoring threads...")
    train_thread = threading.Thread(target=train)
    monitor_thread = threading.Thread(
        target=monitor_checkpoint_directories, args=(output_dir,))

    # wait for training to finish
    train_thread.start()
    monitor_thread.start()

    train_thread.join()
    logging.info("Training thread finished. Stopping monitoring thread...")
    keep_alive = False
    monitor_thread.join()

    logging.info(
        "Training and monitoring threads finished. Uploading Lora")
    upload_final_lora()
