from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
import os
from train_dreambooth_lora_sdxl import import_model_class_from_model_name_or_path

model_name = os.getenv(
    "MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")
vae_path = os.getenv("VAE_PATH", "madebyollin/sdxl-vae-fp16-fix")
variant = "fp16"

# Load the tokenizers
tokenizer_one = AutoTokenizer.from_pretrained(
    model_name,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    model_name,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(
    model_name, None
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    model_name, None, subfolder="text_encoder_2"
)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(
    model_name, subfolder="scheduler")
text_encoder_one = text_encoder_cls_one.from_pretrained(
    model_name, subfolder="text_encoder", revision=None, variant=variant
)
text_encoder_two = text_encoder_cls_two.from_pretrained(
    model_name, subfolder="text_encoder_2", revision=None, variant=variant
)
vae_path = (
    model_name
    if vae_path is None
    else vae_path
)
vae = AutoencoderKL.from_pretrained(
    vae_path,
    subfolder="vae" if vae_path is None else None,
    revision=None,
    variant=None,
)
unet = UNet2DConditionModel.from_pretrained(
    model_name, subfolder="unet", revision=None, variant=variant
)
