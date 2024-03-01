#!/usr/bin/env python3

import os
import requests


#################################################
# These Values Usually Change Every Single Run  #
#################################################
# Replace with your container group name. Remember to increment the
# number for each new container group
container_group_name = "sdxl-timber-lora-12"

# This prefix will be prepended to the file name when files are uploaded.
# e.g. "loras/timber/pytorch_model_weights.safetensors"
checkpoint_bucket_prefix = "loras/timberdog1/"

############################################
# Configuration for the training script    #
# You'll want to play around with these    #
# values to see what works best for your   #
# specific use case.                       #
############################################
prompt = "a photo of timberdog"
max_training_steps = 4000
train_batch_size = 2
learning_rate = "1e-6"
use_8bit_adam = False
mixed_precision = "fp16"
resolution = 1024
gradient_accumulation_steps = 4
lr_scheduler = "constant"
lr_warmup_steps = 0
train_text_encoder = True
gradient_checkpointing = False
with_prior_preservation = False
prior_loss_weight = 1.0


# Should be a script found here:
# https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
training_script = "train_dreambooth_lora_sdxl.py"
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"

# You can automatically log your training runs to Weights & Biases
report_to = "wandb"
wandb_api_key = os.getenv('WANDB_API_KEY', None)
validation_prompt = "a photo of timberdog playing in the snow, pixar style"
validation_epochs = 50  # Run validation every `validation_epochs` epochs

# Checkpoints are saved every `checkpointing_steps` steps
checkpointing_steps = 100


############################################
# Configuration for the container group    #
############################################
salad_api_key = os.getenv('SALAD_API_KEY', None)
if salad_api_key is None:
    raise ValueError('SALAD_API_KEY environment variable not set')

docker_image = "saladtechnologies/dreambooth:sdxl"

# Replace with your organization name
organization_name = "salad-benchmarking"

# Replace with your project name
project_name = "lora-training"

# Replace with the number of vCPUs and amount of memory you want to allocate
num_vcpu = 4  # $0.004/hr/vcpu
memory_gb = 16  # $0.001/hr/gb

# These values can be retrieved from the api:
# https://docs.salad.com/reference/list_gpu_classes
rtx_4090_24gb = "ed563892-aacd-40f5-80b7-90c9be6c759b"  # $0.30/hr
rtx_3090_ti_24gb = "9998fe42-04a5-4807-b3a5-849943f16c38"  # $0.28/hr
rtx_3090_24gb = "a5db5c50-cbcb-4596-ae80-6a0c8090d80f"  # $0.25/hr

# Replace with the GPU class you want to use for the training job
gpu_class = rtx_4090_24gb


############################################
# Configuration for cloud storage          #
############################################
# AWS Config (for S3)
aws_default_region = os.getenv('AWS_DEFAULT_REGION', None)
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', None)
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
aws_endpoint_url = os.getenv('AWS_ENDPOINT_URL', None)

# Replace with your S3 bucket name. This is where the training progress
# will be saved, and where the trained model will be uploaded
checkpoint_bucket_name = "training-checkpoints"


# Replace with your S3 bucket name. This is where the training data
# will be downloaded from
data_bucket_name = "training-data"

# All files in the bucket with this prefix will be downloaded.
# These should all be images in a format that is compatible with Pillow, e.g. jpeg
data_bucket_prefix = "timber/"


###################################################
# Configuration for the training-complete webhook #
###################################################
# When training is complete, a POST webhook will be sent to this URL.
# I recommend configuring this to shut down the container group when training is complete.
# Request payload:
# {
#     "bucket_name": str,
#     "key": str,
#     "machine_id": str,
#     "container_group_id": str,
#     "organization_name": str,
#     "project_name": str,
#     "container_group_name": str
# }
complete_webhook_url = os.getenv('COMPLETE_WEBHOOK_URL', None)

# The header name where an authentication value should be placed, e.g. "x-api-key"
complete_webhook_auth_header = os.getenv('COMPLETE_WEBHOOK_AUTH_HEADER', None)

# The value that should be placed in the header specified by `complete_webhook_auth_header`
complete_webhook_auth_value = os.getenv('COMPLETE_WEBHOOK_AUTH_VALUE', None)


###########################################################
# Create the container group using the Salad API          #
# https://docs.salad.com/reference/create_container_group #
###########################################################
url = f"https://api.salad.com/api/public/organizations/{organization_name}/projects/{project_name}/containers"
payload = {
    "name": container_group_name,
    "container": {
        "image": docker_image,
        "resources": {
            "cpu": num_vcpu,
            "memory": memory_gb * 1024,
            "gpu_classes": [gpu_class],
            "storage_amount": 5 * 1024 * 1024 * 1024  # 5GB
        },
        "environment_variables": {
            "PROMPT": prompt,
            "LEARNING_RATE": str(learning_rate),
            "MAX_TRAIN_STEPS": str(max_training_steps),
            "TRAIN_BATCH_SIZE": str(train_batch_size),
            "USE_8BIT_ADAM": str(use_8bit_adam).lower(),
            "MIXED_PRECISION": mixed_precision,
            "RESOLUTION": str(resolution),
            "GRADIENT_ACCUMULATION_STEPS": str(gradient_accumulation_steps),
            "LR_SCHEDULER": lr_scheduler,
            "LR_WARMUP_STEPS": str(lr_warmup_steps),
            "TRAIN_TEXT_ENCODER": str(train_text_encoder).lower(),
            "GRADIENT_CHECKPOINTING": str(gradient_checkpointing).lower(),
            "MODEL_NAME": model_name,
            "VAE_PATH": vae_path,
            "TRAINING_SCRIPT": training_script,
            "CHECKPOINTING_STEPS": str(checkpointing_steps),
            "SALAD_ORGANIZATION_NAME": organization_name,
            "SALAD_PROJECT_NAME": project_name,
            "SALAD_CONTAINER_GROUP_NAME": container_group_name,
        },
    },
    "autostart_policy": True,
    "replicas": 1
}

if with_prior_preservation:
    payload["container"]["environment_variables"]['WITH_PRIOR_PRESERVATION'] = "true"
    payload["container"]["environment_variables"]['PRIOR_LOSS_WEIGHT'] = str(
        prior_loss_weight)

if report_to == "wandb":
    payload["container"]["environment_variables"]['REPORT_TO'] = "wandb"
    payload["container"]["environment_variables"]['WANDB_API_KEY'] = wandb_api_key

if report_to is not None and validation_prompt is not None:
    payload["container"]["environment_variables"]["VALIDATION_PROMPT"] = validation_prompt
    payload["container"]["environment_variables"]["VALIDATION_EPOCHS"] = str(
        validation_epochs)

if aws_access_key_id is not None:
    payload["container"]["environment_variables"]["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    payload["container"]["environment_variables"]["CHECKPOINT_BUCKET_NAME"] = checkpoint_bucket_name
    payload["container"]["environment_variables"]["CHECKPOINT_BUCKET_PREFIX"] = checkpoint_bucket_prefix
    payload["container"]["environment_variables"]["DATA_BUCKET_NAME"] = data_bucket_name
    payload["container"]["environment_variables"]["DATA_BUCKET_PREFIX"] = data_bucket_prefix

if aws_secret_access_key is not None:
    payload["container"]["environment_variables"]["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

if aws_default_region is not None:
    payload["container"]["environment_variables"]["AWS_DEFAULT_REGION"] = aws_default_region

if aws_endpoint_url is not None:
    payload["container"]["environment_variables"]["AWS_ENDPOINT_URL"] = aws_endpoint_url

if complete_webhook_url is not None:
    payload["container"]["environment_variables"]["COMPLETE_WEBHOOK_URL"] = complete_webhook_url

if complete_webhook_auth_header is not None:
    payload["container"]["environment_variables"]["COMPLETE_WEBHOOK_AUTH_HEADER"] = complete_webhook_auth_header
    payload["container"]["environment_variables"]["COMPLETE_WEBHOOK_AUTH_VALUE"] = complete_webhook_auth_value

print(f"POST {url}")

response = requests.post(
    url,
    headers={"Salad-Api-Key": salad_api_key},
    json=payload
)

print(response.status_code, response.reason)
if not response.ok:
    print(response.json())
else:
    print(
        f"Container group {container_group_name} created successfully! View in Portal:")
    print(
        f"https://portal.salad.com/organizations/{organization_name}/projects/{project_name}/containers/{container_group_name}")
