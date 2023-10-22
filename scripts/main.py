from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid
from lcm.lcm_scheduler import LCMScheduler
from lcm.lcm_pipeline import LatentConsistencyModelPipeline
import modules.scripts as scripts
from modules import script_callbacks
import os
import random
import time
import numpy as np
import gradio as gr
from PIL import Image, PngImagePlugin
import torch


DESCRIPTION = '''# Latent Consistency Model
Distilled from [Dreamshaper v7](https://huggingface.co/Lykon/dreamshaper-7) fine-tune of [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with only 4,000 training iterations (~32 A100 GPU Hours). [Project page](https://latent-consistency-models.github.io)
'''

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "LCM"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_image(img, metadata: dict):
    save_dir = os.path.join(scripts.basedir(), "outputs/txt2img-images/LCM/")
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    seed = metadata["seed"]
    unique_id = uuid.uuid4()
    filename = save_dir + f"{unique_id}-{seed}" + ".png"

    meta_tuples = [(k, str(v)) for k, v in metadata.items()]
    png_info = PngImagePlugin.PngInfo()
    for k, v in meta_tuples:
        png_info.add_text(k, v)
    img.save(filename, pnginfo=png_info)

    return filename


def save_images(image_array, metadata: dict):
    paths = []
    with ThreadPoolExecutor() as executor:
        paths = list(executor.map(save_image, image_array,
                     [metadata]*len(image_array)))
    return paths


scheduler = LCMScheduler.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
pipe = LatentConsistencyModelPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler)
pipe.safety_checker = None  # ¯\_(ツ)_/¯
pipe.to("cuda")


def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        lcm_origin_steps=50,
        output_type="pil",
    ).images
    paths = save_images(result, metadata={"prompt": prompt, "seed": seed, "width": width,
                        "height": height, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps})
    print("LCM inference time: ", time.time() - start_time, "seconds")
    return paths, seed


examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]


def on_ui_tabs():
    with gr.Blocks(css="style.css") as lcm:
        gr.Markdown(DESCRIPTION)
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery", grid=[2], preview=True
            )
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
                randomize=True
            )
            randomize_seed = gr.Checkbox(
                label="Randomize seed across runs", value=True)
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale for base",
                    minimum=2,
                    maximum=14,
                    step=0.1,
                    value=8.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps for base",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )
            with gr.Row():
                num_images = gr.Slider(
                    label="Number of images",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                    visible=False,
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            fn=generate,
            cache_examples=CACHE_EXAMPLES,
        )

        run_button.click(
            fn=generate,
            inputs=[
                prompt,
                seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                num_images,
                randomize_seed
            ],
            outputs=[result, seed],
        )
    return [(lcm, "LCM", "lcm")]


script_callbacks.on_ui_tabs(on_ui_tabs)
