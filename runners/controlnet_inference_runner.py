import os
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import AutoTokenizer, CLIPTextModel
import torch
from PIL import Image


def controlnet_inference_runner(args):

    # Set weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    condition_image = Image.open(args.condition_image).convert("RGB")
    
    # Load ControlNet and Stable Diffusion models
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=weight_dtype)
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_model, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.sd_model, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.sd_model, subfolder="unet", revision=args.revision, variant=args.variant
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.sd_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
    )

    # Configure pipeline
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if args.use_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Generate image
    generator = torch.manual_seed(args.seed)
    image = pipe(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        image=condition_image,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt
    ).images[0]
    
    # Save the generated image
    os.makedirs(args.output_path, exist_ok=True)
    image.save(os.path.join(args.output_path, "generated_image.png"))
    print(f"Generated image saved as {os.path.join(args.output_path, 'generated_image.png')}")