# Dreambooth LoRA Training

## Environment Variables

| Variable Name                | Default Value                            | Description                               |
| ---------------------------- | ---------------------------------------- | ----------------------------------------- |
| LOG_LEVEL                    | INFO                                     | Log level configuration                   |
| MODEL_NAME                   | stabilityai/stable-diffusion-xl-base-1.0 | Huggingface Hub Model Name or Path        |
| INSTANCE_DIR                 | /images                                  | Directory where training data is stored   |
| OUTPUT_DIR                   | /output                                  | Directory where training output is stored |
| VAE_PATH                     | madebyollin/sdxl-vae-fp16-fix            | VAE model name or path                    |
| PROMPT                       | photo of timberdog                       | Prompt for training                       |
| DREAMBOOTH_SCRIPT            | train_dreambooth_lora_sdxl.py            | Dreambooth training script path           |
| RESOLUTION                   | 1024                                     | Resolution of the images                  |
| MAX_TRAIN_STEPS              | 500                                      | Total number of training steps            |
| CHECKPOINTING_STEPS          | 50                                       | Save a checkpoint after every N steps     |
| LEARNING_RATE                | 1e-4                                     | Learning rate                             |
| GRADIENT_ACCUMULATION_STEPS  | 4                                        | Gradient accumulation steps               |
| LR_WARMUP_STEPS              | 0                                        | LR warmup steps                           |
| MIXED_PRECISION              | fp16                                     | Mixed precision training                  |
| TRAIN_BATCH_SIZE             | 1                                        | Train batch size                          |
| LR_SCHEDULER                 | constant                                 | Learning rate scheduler                   |
| USE_8BIT_ADAM                | None                                     | Use 8-bit adam                            |
| TRAIN_TEXT_ENCODER           | None                                     | Train text encoder                        |
| GRADIENT_CHECKPOINTING       | None                                     | Gradient checkpointing                    |
| WITH_PRIOR_PRESERVATION      | None                                     | With prior preservation                   |
| PRIOR_LOSS_WEIGHT            | 1.0                                      | Prior loss weight                         |
| CHECKPOINT_BUCKET_NAME       | None                                     | S3 bucket name for storing checkpoints    |
| CHECKPOINT_BUCKET_PREFIX     | None                                     | Prefix for storing checkpoints in S3      |
| DATA_BUCKET_NAME             | None                                     | S3 bucket name for storing training data  |
| DATA_BUCKET_PREFIX           | None                                     | Prefix for storing training data in S3    |
| WEBHOOK_URL                  | None                                     | Webhook URL                               |
| PROGRESS_WEBHOOK_URL         | None                                     | Webhook URL for progress                  |
| COMPLETE_WEBHOOK_URL         | None                                     | Webhook URL for completion                |
| WEBHOOK_AUTH_HEADER          | None                                     | Authentication header for webhook         |
| PROGRESS_WEBHOOK_AUTH_HEADER | None                                     | Auth header for progress webhook          |
| COMPLETE_WEBHOOK_AUTH_HEADER | None                                     | Auth header for completion webhook        |
| WEBHOOK_AUTH_VALUE           | None                                     | Authentication value for webhook          |
| PROGRESS_WEBHOOK_AUTH_VALUE  | None                                     | Auth value for progress webhook           |
| COMPLETE_WEBHOOK_AUTH_VALUE  | None                                     | Auth value for completion webhook         |
| SALAD_MACHINE_ID             | None                                     | Salad Machine ID                          |
| SALAD_CONTAINER_GROUP_ID     | None                                     | Container Group ID for Salad              |
| SALAD_CONTAINER_GROUP_NAME   | None                                     | Container Group name for Salad            |
| SALAD_ORGANIZATION_NAME      | None                                     | Organization name for Salad               |
| SALAD_PROJECT_NAME           | None                                     | Project name for Salad                    |
