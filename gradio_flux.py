import gradio as gr
import os
import math
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from src.pipeline_flux_tryon import FluxPipeline
from src.transformer_flux_garm import FluxTransformer2DModel as FluxTransformer2DModel_Garm
from src.transformer_flux_vton import FluxTransformer2DModel as FluxTransformer2DModel_Vton
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
from src.pose_guider import PoseGuider
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
from huggingface_hub import hf_hub_download

example_path = os.path.join(os.path.dirname(__file__), 'examples')

class FluxFitGenerator:
    def __init__(self, flux_path, fluxfit_path, device, offload, aggressive_offload, revision):
        transformer_garm = FluxTransformer2DModel_Garm.from_pretrained(flux_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        transformer_vton = FluxTransformer2DModel_Vton.from_pretrained(fluxfit_path, subfolder="transformer_vton", revision=revision, torch_dtype=torch.bfloat16)
        pose_guider =  PoseGuider(conditioning_embedding_channels=3072, conditioning_channels=3, block_out_channels=(16, 32, 96, 256))
        pose_guider.load_state_dict(torch.load(hf_hub_download(repo_id=fluxfit_path, filename="pose_guider/diffusion_pytorch_model.bin", revision=revision)))
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        pose_guider.to(dtype=torch.bfloat16)
        image_encoder.to(dtype=torch.bfloat16)
        self.pipeline = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16, transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, image_encoder=image_encoder)
        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(device='cpu')
            self.parsing_model = Parsing(device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(device='cpu')
            self.parsing_model = Parsing(device='cpu')
        else:
            self.pipeline.to(device)
            self.dwprocessor = DWposeDetector(device)
            self.parsing_model = Parsing(device)
        

    def process(self, vton_img, garm_img, category, n_steps, image_scale, seed, prompt):
        with torch.inference_mode():
            garm_img = Image.open(garm_img)
            vton_img = Image.open(vton_img)
            new_width = 768
            new_height = 1024

            garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img.width
            candidate[:, 1]*=vton_img.height

            pose_image = pose_image[:,:,::-1] #rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, width=model_parse.width, height=model_parse.height)

            if category=="Upper_body":
                cloth_prompt = "a photo of upper body garment."
            elif category=="Lower_body":
                cloth_prompt = "a photo of lower body garment."
            else:
                cloth_prompt = "a photo of dresses."
            res = self.pipeline(
                prompt=prompt,
                cloth_prompt=cloth_prompt,
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                max_sequence_length=77,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
            ).images[0]
            return res


def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))

    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

HEADER = """
<h1 style="text-align: center;"> FluxFit: Virtual Fitting based on Flux </h1>
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://github.com/BoyuanJiang/FluxFit" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href='https://huggingface.co/BoyuanJiang/FluxFit/tree/main' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="http://demo.fluxfit.byjiang.com/" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
</div>
<br>
· The first work to expand the powerful Flux model to virtual fitting task. It can only be used for Non-commercial Use. <br>
· The effect is not ideal for some complex textured clothing and model with complex limbs, we are still trying to optimize it.
"""

def create_demo(flux_path, fluxfit_path, device, offload, aggressive_offload, revision):
    generator = FluxFitGenerator(flux_path, fluxfit_path, device, offload, aggressive_offload, revision)
    with gr.Blocks(title="FluxFit") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column():
                vton_img = gr.Image(label="Model", sources=None, type="filepath")
                example = gr.Examples(
                    label="Examples (upper-body/lower-body)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/0.jpg'),
                        os.path.join(example_path, 'model/1.jpg'),
                        os.path.join(example_path, 'model/2.jpg'),
                        os.path.join(example_path, 'model/3.png'),
                    ])
                example = gr.Examples(
                    label="Examples (dress)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/4.jpg'),
                        os.path.join(example_path, 'model/5.jpg'),
                        os.path.join(example_path, 'model/6.jpg'),
                        os.path.join(example_path, 'model/7.jpg'),
                    ])
            with gr.Column():
                garm_img = gr.Image(label="Garment", sources=None, type="filepath")
                category = gr.Dropdown(label="Garment category", choices=["Upper-body", "Lower-body", "Dresses"], value="Upper-body")
                example = gr.Examples(
                    label="Examples (upper-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/0.jpg'),
                        os.path.join(example_path, 'garment/1.jpg'),
                        os.path.join(example_path, 'garment/2.jpg'),
                        os.path.join(example_path, 'garment/3.jpg'),
                    ])
                example = gr.Examples(
                    label="Examples (lower-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/4.jpg'),
                        os.path.join(example_path, 'garment/5.jpg'),
                        os.path.join(example_path, 'garment/6.jpeg'),
                        os.path.join(example_path, 'garment/7.jpg'),
                    ])
                example = gr.Examples(
                    label="Examples (dress)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/8.jpg'),
                        os.path.join(example_path, 'garment/9.png'),
                        os.path.join(example_path, 'garment/10.jpg'),
                        os.path.join(example_path, 'garment/11.jpg'),
                    ])
            with gr.Column():
                result_gallery = gr.Image(label="Output", elem_id="output-img",show_share_button=False)
        with gr.Column():
            run_button = gr.Button(value="Run")
            n_steps = gr.Slider(label="Steps", minimum=10, maximum=20, value=15, step=1)
            image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=3.5, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
            prompt = gr.Textbox(placeholder="Describe the garment the model is fitting. It's optional but with accurate description will achieve better results.", show_label=False, elem_id="prompt")
        with gr.Row():
            gr.HTML("""
            <div id="clustrmaps" style="width: 100%; height: 200px; display: flex; justify-content: center; align-items: center;">
            <a href='https://clustrmaps.com/site/1c1nb'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=7ecbbd&w=a&t=tt&d=LID0oYYSmutyFLdphIiI4z8jAuqIpvQxIAcaMumfnlc&co=ffffff&ct=808080'/></a>
            </div>
                """)

        ips_dc = [vton_img, garm_img, category, n_steps, image_scale, seed, prompt]
        run_button.click(fn=generator.process, inputs=ips_dc, outputs=[result_gallery])
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FluxFit")
    parser.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev", help="The path of Flux.1 dev model.")
    parser.add_argument("--fluxfit_path", type=str, default="BoyuanJiang/FluxFit", help="The path of FluxFit model.")
    parser.add_argument("--revision", type=str, default="v1.0", help="FluxFit model version.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use, can work under 32G GPU memory")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, can work under 8G GPU memory")
    args = parser.parse_args()
    demo = create_demo(args.flux_path, args.fluxfit_path, args.device, args.offload, args.aggressive_offload, args.revision)
    demo.launch(share=True)
