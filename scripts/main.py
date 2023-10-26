from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import uuid
from lcm.lcm_scheduler import LCMScheduler
from lcm.lcm_pipeline import LatentConsistencyModelPipeline
from lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
from diffusers.image_processor import PipelineImageInput
import modules.scripts as scripts
import modules.shared
from modules import script_callbacks
import os
import random
import time
import numpy as np
import gradio as gr
from PIL import Image, PngImagePlugin
import torch


DESCRIPTION = '''# Latent Consistency Model
Running [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | [Project Page](https://latent-consistency-models.github.io) | [Extension Page](https://github.com/0xbitches/sd-webui-lcm)
'''

MAX_SEED = np.iinfo(np.int32).max
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


def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    use_fp16: bool = True,
    use_torch_compile: bool = False,
    use_cpu: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)

    selected_device = modules.shared.device
    if use_cpu:
        selected_device = "cpu"
        if use_fp16:
            use_fp16 = False
            print("LCM warning: running on CPU, overrode FP16 with FP32")

    scheduler = LCMScheduler.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
    pipe = LatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", scheduler = scheduler, safety_checker = None)

    if use_fp16:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float16)
    else:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float32)

    # Windows does not support torch.compile for now
    if os.name != 'nt' and use_torch_compile:
        pipe.unet = torch.compile(pipe.unet, mode='max-autotune')

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        original_inference_steps=50,
        output_type="pil",
        device = selected_device
    ).images
    paths = save_images(result, metadata={"prompt": prompt, "seed": seed, "width": width,
                        "height": height, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps})

    elapsed_time = time.time() - start_time
    print("LCM inference time: ", elapsed_time, "seconds")
    return paths, seed


def generate_i2i(
    prompt: str,
    image: PipelineImageInput = None,
    strength: float = 0.8,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    use_fp16: bool = True,
    use_torch_compile: bool = False,
    use_cpu: bool = False,
    progress=gr.Progress(track_tqdm=True),
    width: Optional[int] = 512,
    height: Optional[int] = 512,
) -> Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)

    selected_device = modules.shared.device
    if use_cpu:
        selected_device = "cpu"
        if use_fp16:
            use_fp16 = False
            print("LCM warning: running on CPU, overrode FP16 with FP32")

    pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", safety_checker = None)

    if use_fp16:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float16)
    else:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float32)

    # Windows does not support torch.compile for now
    if os.name != 'nt' and use_torch_compile:
        pipe.unet = torch.compile(pipe.unet, mode='max-autotune')

    width, height = image.size

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        original_inference_steps=50,
        output_type="pil",
        device = selected_device
    ).images
    paths = save_images(result, metadata={"prompt": prompt, "seed": seed, "width": width,
                        "height": height, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps})

    elapsed_time = time.time() - start_time
    print("LCM inference time: ", elapsed_time, "seconds")
    return paths, seed

import cv2

def video_to_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: LCM Could not open video.")
        return
    
    # Read frames from the video
    pil_images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Append the PIL Image to the list
        pil_images.append(pil_image)
    
    # Release the video capture object
    cap.release()
    
    return pil_images

def frames_to_video(pil_images, output_path, fps):
    if not pil_images:
        print("Error: No images to convert.")
        return
    
    img_array = []
    for pil_image in pil_images:
        img_array.append(np.array(pil_image))
    
    height, width, layers = img_array[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR))
    out.release()

def generate_v2v(
    prompt: str,
    video: any = None,
    strength: float = 0.8,
    seed: int = 0,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    randomize_seed: bool = False,
    use_fp16: bool = True,
    use_torch_compile: bool = False,
    use_cpu: bool = False,
    fps: int = 10,
    save_frames: bool = False,
    # progress=gr.Progress(track_tqdm=True),
    width: Optional[int] = 512,
    height: Optional[int] = 512,
    num_images: Optional[int] = 1,
) -> Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)

    selected_device = modules.shared.device
    if use_cpu:
        selected_device = "cpu"
        if use_fp16:
            use_fp16 = False
            print("LCM warning: running on CPU, overrode FP16 with FP32")

    pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7", safety_checker = None)

    if use_fp16:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float16)
    else:
        pipe.to(torch_device=selected_device, torch_dtype=torch.float32)

    # Windows does not support torch.compile for now
    if os.name != 'nt' and use_torch_compile:
        pipe.unet = torch.compile(pipe.unet, mode='max-autotune')

    frames = video_to_frames(video)
    if frames is None:
        print("Error: LCM could not convert video.")
        return
    width, height = frames[0].size

    start_time = time.time()

    results = []
    for frame in frames:
        result = pipe(
            prompt=prompt,
            image=frame,
            strength=strength,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            original_inference_steps=50,
            output_type="pil",
            device = selected_device
        ).images
        if save_frames:
            paths = save_images(result, metadata={"prompt": prompt, "seed": seed, "width": width,
                                "height": height, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps})
        results.extend(result)

    elapsed_time = time.time() - start_time
    print("LCM vid2vid inference complete! Processing", len(frames), "frames took", elapsed_time, "seconds")
    
    save_dir = os.path.join(scripts.basedir(), "outputs/LCM-vid2vid/")
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    unique_id = uuid.uuid4()
    _, input_ext = os.path.splitext(video)
    output_path = save_dir + f"{unique_id}-{seed}" + f"{input_ext}"
    frames_to_video(results, output_path, fps)
    return output_path



examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]


def on_ui_tabs():
    with gr.Blocks() as lcm:
        with gr.Tab("LCM txt2img"):
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", 
                                    show_label=False, 
                                    lines=3, 
                                    placeholder="Prompt", 
                                    elem_classes=["prompt"])     
                run_button = gr.Button("Run", scale=0)
            with gr.Row():        
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
                use_fp16 = gr.Checkbox(
                    label="Run LCM in fp16 (for lower VRAM)", value=True)
                use_torch_compile = gr.Checkbox(
                    label="Run LCM with torch.compile (currently not supported on Windows)", value=False)
                use_cpu = gr.Checkbox(label="Run LCM on CPU", value=False)
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
                        label="Number of images (batch count)",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=4,
                    )

            gr.Examples(
                examples=examples,
                inputs=prompt,
                outputs=result,
                fn=generate
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
                    randomize_seed,
                    use_fp16,
                    use_torch_compile,
                    use_cpu
                ],
                outputs=[result, seed],
            )
    
        with gr.Tab("LCM img2img"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", 
                                    show_label=False, 
                                    lines=3, 
                                    placeholder="Prompt", 
                                    elem_classes=["prompt"])       
                run_i2i_button = gr.Button("Run", scale=0)
            with gr.Row():      
                image_input = gr.Image(label="Upload your Image", type="pil")
                result = gr.Gallery(
                    label="Generated images", 
                    show_label=False, 
                    elem_id="gallery", 
                    preview=True
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
                use_fp16 = gr.Checkbox(
                    label="Run LCM in fp16 (for lower VRAM)", value=True)
                use_torch_compile = gr.Checkbox(
                    label="Run LCM with torch.compile (currently not supported on Windows)", value=False)
                use_cpu = gr.Checkbox(label="Run LCM on CPU", value=False)
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
                        label="Number of images (batch count)",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=4,
                    )
                    strength = gr.Slider(
                        label="Prompt Strength",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                    )

            run_i2i_button.click(
                fn=generate_i2i,
                inputs=[
                    prompt,
                    image_input,
                    strength,
                    seed,
                    guidance_scale,
                    num_inference_steps,
                    num_images,
                    randomize_seed,
                    use_fp16,
                    use_torch_compile,
                    use_cpu
                ],
                outputs=[result, seed],
            )
        
        
        with gr.Tab("LCM vid2vid"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", 
                                    show_label=False, 
                                    lines=3, 
                                    placeholder="Prompt", 
                                    elem_classes=["prompt"])       
                run_v2v_button = gr.Button("Run", scale=0)
            with gr.Row():
                video_input = gr.Video(label="Source Video")
                video_output = gr.Video(label="Generated Video")

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
                use_fp16 = gr.Checkbox(
                    label="Run LCM in fp16 (for lower VRAM)", value=True)
                use_torch_compile = gr.Checkbox(
                    label="Run LCM with torch.compile (currently not supported on Windows)", value=False)
                use_cpu = gr.Checkbox(label="Run LCM on CPU", value=False)
                save_frames = gr.Checkbox(label="Save intermediate frames", value=False)                   
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
                    fps = gr.Slider(
                        label="Output FPS",
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=10,
                    )
                    strength = gr.Slider(
                        label="Prompt Strength",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                    )

            run_v2v_button.click(
                fn=generate_v2v,
                inputs=[
                    prompt,
                    video_input,
                    strength,
                    seed,
                    guidance_scale,
                    num_inference_steps,
                    randomize_seed,
                    use_fp16,
                    use_torch_compile,
                    use_cpu,
                    fps,
                    save_frames
                ],
                outputs=video_output,
            )
        
        

    return [(lcm, "LCM", "lcm")]



script_callbacks.on_ui_tabs(on_ui_tabs)