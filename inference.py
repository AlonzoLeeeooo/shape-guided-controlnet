import argparse

from runners.controlnet_inference_runner import controlnet_inference_runner

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'
                          
def parse_args():
    parser = argparse.ArgumentParser(description="Shape-guided ControlNet inference")
    parser.add_argument("--condition_image", type=str, default="", help="Local path to condition image")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny", help="ControlNet model name or path")
    parser.add_argument("--sd_model", type=str, default="", help="Stable Diffusion model name or path")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for generation")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale for image generation")
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision for image generation")
    parser.add_argument("--output_path", type=str, default="", help="Local path to save the generated image")
    parser.add_argument("--revis3ion", type=str, default="main", help="Revision of the model")
    parser.add_argument("--variant", type=str, default=None, choices=["fp16", "bf16"], help="Variant of the model")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for image generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controlnet_inference_runner(args)