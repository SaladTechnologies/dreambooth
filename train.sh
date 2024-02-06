#! /usr/bin/env bash

autotrain dreambooth \
--model stabilityai/stable-diffusion-xl-base-1.0 \
--image-path /images \
--prompt "timber boy" \
--resolution 1024 \
--batch-size 1 \
--checkpointing-steps 50 \
--resume-from-checkpoint latest \
--checkpoints-total-limit 1 \
--num-steps 500 \
--fp16 \
--gradient-accumulation 4 \
--lr 1e-4 \
--project-name "/output/timber"