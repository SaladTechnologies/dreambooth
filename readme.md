# Dreambooth LoRA Training

## Environment Variables

| Variable Name                | Default Value                            | Description                                         |
| ---------------------------- | ---------------------------------------- | --------------------------------------------------- |
| LOG_LEVEL                    | INFO                                     | Log level for the application                       |
| MODEL_NAME                   | stabilityai/stable-diffusion-xl-base-1.0 | Name of the model used                              |
| INSTANCE_DIR                 | /images                                  | Directory for instances                             |
| OUTPUT_DIR                   | /output                                  | Directory for output files                          |
| VAE_PATH                     | madebyollin/sdxl-vae-fp16-fix            | Path to the VAE model                               |
| PROMPT                       | photo of timberdog                       | Prompt for the application                          |
| CHECKPOINT_BUCKET_NAME       | None                                     | Name of the bucket for checkpoints (if applicable)  |
| CHECKPOINT_BUCKET_PREFIX     | None                                     | Prefix for the checkpoint bucket (if applicable)    |
| DATA_BUCKET_NAME             | None                                     | Name of the bucket for data storage (if applicable) |
| DATA_BUCKET_PREFIX           | None                                     | Prefix for the data bucket (if applicable)          |
| WEBHOOK_URL                  | None                                     | URL for webhooks                                    |
| PROGRESS_WEBHOOK_URL         | Value of 'WEBHOOK_URL'                   | URL for progress webhooks                           |
| COMPLETE_WEBHOOK_URL         | Value of 'WEBHOOK_URL'                   | URL for complete webhooks                           |
| WEBHOOK_AUTH_HEADER          | None                                     | Authorization header for webhooks                   |
| PROGRESS_WEBHOOK_AUTH_HEADER | Value of 'WEBHOOK_AUTH_HEADER'           | Authorization header for progress webhooks          |
| COMPLETE_WEBHOOK_AUTH_HEADER | Value of 'WEBHOOK_AUTH_HEADER'           | Authorization header for complete webhooks          |
| WEBHOOK_AUTH_VALUE           | None                                     | Authorization value for webhooks                    |
| PROGRESS_WEBHOOK_AUTH_VALUE  | Value of 'WEBHOOK_AUTH_VALUE'            | Authorization value for progress webhooks           |
| COMPLETE_WEBHOOK_AUTH_VALUE  | Value of 'WEBHOOK_AUTH_VALUE'            | Authorization value for complete webhooks           |
| SALAD_MACHINE_ID             | None                                     | ID for the salad machine                            |
| SALAD_CONTAINER_GROUP_ID     | None                                     | ID for the salad container group                    |
