# AutoTrain Advanced

## Dreambooth Options

```
usage: autotrain <command> [<args>] dreambooth [-h] [--revision REVISION]
                                               [--tokenizer TOKENIZER]
                                               --image-path IMAGE_PATH
                                               [--class-image-path CLASS_IMAGE_PATH]
                                               --prompt PROMPT
                                               [--class-prompt CLASS_PROMPT]
                                               [--num-class-images NUM_CLASS_IMAGES]
                                               [--class-labels-conditioning CLASS_LABELS_CONDITIONING]
                                               [--prior-preservation]
                                               [--prior-loss-weight PRIOR_LOSS_WEIGHT]
                                               --resolution RESOLUTION
                                               [--center-crop]
                                               [--train-text-encoder]
                                               [--sample-batch-size SAMPLE_BATCH_SIZE]
                                               [--num-steps NUM_STEPS]
                                               [--checkpointing-steps CHECKPOINTING_STEPS]
                                               [--resume-from-checkpoint RESUME_FROM_CHECKPOINT]
                                               [--scale-lr]
                                               [--scheduler SCHEDULER]
                                               [--warmup-steps WARMUP_STEPS]
                                               [--num-cycles NUM_CYCLES]
                                               [--lr-power LR_POWER]
                                               [--dataloader-num-workers DATALOADER_NUM_WORKERS]
                                               [--use-8bit-adam]
                                               [--adam-beta1 ADAM_BETA1]
                                               [--adam-beta2 ADAM_BETA2]
                                               [--adam-weight-decay ADAM_WEIGHT_DECAY]
                                               [--adam-epsilon ADAM_EPSILON]
                                               [--max-grad-norm MAX_GRAD_NORM]
                                               [--allow-tf32]
                                               [--prior-generation-precision PRIOR_GENERATION_PRECISION]
                                               [--local-rank LOCAL_RANK]
                                               [--xformers]
                                               [--pre-compute-text-embeddings]
                                               [--tokenizer-max-length TOKENIZER_MAX_LENGTH]
                                               [--text-encoder-use-attention-mask]
                                               [--rank RANK] [--xl] [--fp16]
                                               [--bf16]
                                               [--validation-prompt VALIDATION_PROMPT]
                                               [--num-validation-images NUM_VALIDATION_IMAGES]
                                               [--validation-epochs VALIDATION_EPOCHS]
                                               [--checkpoints-total-limit CHECKPOINTS_TOTAL_LIMIT]
                                               [--validation-images VALIDATION_IMAGES]
                                               [--logging] [--train]
                                               [--deploy] [--inference]
                                               [--username USERNAME]
                                               [--backend BACKEND]
                                               [--token TOKEN]
                                               [--repo-id REPO_ID]
                                               [--push-to-hub] --model MODEL
                                               --project-name PROJECT_NAME
                                               [--seed SEED] [--epochs EPOCHS]
                                               [--gradient-accumulation GRADIENT_ACCUMULATION]
                                               [--disable_gradient_checkpointing]
                                               [--lr LR] [--log LOG]
                                               [--data-path DATA_PATH]
                                               [--train-split TRAIN_SPLIT]
                                               [--valid-split VALID_SPLIT]
                                               [--batch-size BATCH_SIZE]

✨ Run AutoTrain DreamBooth Training

options:
  -h, --help            show this help message and exit
  --revision REVISION   Model revision to use for training
  --tokenizer TOKENIZER
                        Tokenizer to use for training
  --image-path IMAGE_PATH
                        Path to the images
  --class-image-path CLASS_IMAGE_PATH
                        Path to the class images
  --prompt PROMPT       Instance prompt
  --class-prompt CLASS_PROMPT
                        Class prompt
  --num-class-images NUM_CLASS_IMAGES
                        Number of class images
  --class-labels-conditioning CLASS_LABELS_CONDITIONING
                        Class labels conditioning
  --prior-preservation  With prior preservation
  --prior-loss-weight PRIOR_LOSS_WEIGHT
                        Prior loss weight
  --resolution RESOLUTION
                        Resolution
  --center-crop         Center crop
  --train-text-encoder  Train text encoder
  --sample-batch-size SAMPLE_BATCH_SIZE
                        Sample batch size
  --num-steps NUM_STEPS
                        Max train steps
  --checkpointing-steps CHECKPOINTING_STEPS
                        Checkpointing steps
  --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                        Resume from checkpoint
  --scale-lr            Scale learning rate
  --scheduler SCHEDULER
                        Learning rate scheduler
  --warmup-steps WARMUP_STEPS
                        Learning rate warmup steps
  --num-cycles NUM_CYCLES
                        Learning rate num cycles
  --lr-power LR_POWER   Learning rate power
  --dataloader-num-workers DATALOADER_NUM_WORKERS
                        Dataloader num workers
  --use-8bit-adam       Use 8bit adam
  --adam-beta1 ADAM_BETA1
                        Adam beta 1
  --adam-beta2 ADAM_BETA2
                        Adam beta 2
  --adam-weight-decay ADAM_WEIGHT_DECAY
                        Adam weight decay
  --adam-epsilon ADAM_EPSILON
                        Adam epsilon
  --max-grad-norm MAX_GRAD_NORM
                        Max grad norm
  --allow-tf32          Allow TF32
  --prior-generation-precision PRIOR_GENERATION_PRECISION
                        Prior generation precision
  --local-rank LOCAL_RANK
                        Local rank
  --xformers            Enable xformers memory efficient attention
  --pre-compute-text-embeddings
                        Pre compute text embeddings
  --tokenizer-max-length TOKENIZER_MAX_LENGTH
                        Tokenizer max length
  --text-encoder-use-attention-mask
                        Text encoder use attention mask
  --rank RANK           Rank
  --xl                  XL
  --fp16                FP16
  --bf16                BF16
  --validation-prompt VALIDATION_PROMPT
                        Validation prompt
  --num-validation-images NUM_VALIDATION_IMAGES
                        Number of validation images
  --validation-epochs VALIDATION_EPOCHS
                        Validation epochs
  --checkpoints-total-limit CHECKPOINTS_TOTAL_LIMIT
                        Checkpoints total limit
  --validation-images VALIDATION_IMAGES
                        Validation images
  --logging             Logging using tensorboard
  --train               Train the model
  --deploy              Deploy the model
  --inference           Run inference
  --username USERNAME   Hugging Face Hub Username
  --backend BACKEND     Backend to use: default or spaces. Spaces backend
                        requires push_to_hub and repo_id
  --token TOKEN         Hub token
  --repo-id REPO_ID     Hub repo id
  --push-to-hub         Push to hub
  --model MODEL         Model to use for training
  --project-name PROJECT_NAME
                        Output directory or repo id
  --seed SEED           Seed
  --epochs EPOCHS       Number of training epochs
  --gradient-accumulation GRADIENT_ACCUMULATION
                        Gradient accumulation steps
  --disable_gradient_checkpointing
                        Disable gradient checkpointing
  --lr LR               Learning rate
  --log LOG             Use experiment tracking
  --data-path DATA_PATH
                        Train dataset to use
  --train-split TRAIN_SPLIT
                        Test dataset split to use
  --valid-split VALID_SPLIT
                        Validation dataset split to use
  --batch-size BATCH_SIZE
                        Training batch size to use
```