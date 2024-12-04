<h1 style="text-align: center; color: #ff0000; font-family: Arial, sans-serif;">
    This repository will no longer be maintained. 
    For subsequent updates, please refer to 
    <a href="https://github.com/BoyuanJiang/FitDiT" style="color: #007bff; text-decoration: none;">FitDiT</a>.
</h1>

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
· The first work to extend the powerful Flux model to virtual fitting task. It can only be used for Non-commercial Use. <br>
· The effect is not ideal for some complex textured clothing and model with complex limbs, we are still trying to optimize it. <br>

# Updates
- **`2024/10/12`**: Our [**Online Demo**](http://demo.fluxfit.byjiang.com/) with v1.0 checkpoint is released.

# Web Demo
Running FluxFit consumes a lot of GPU, we recommend using the [online demo](http://demo.fluxfit.byjiang.com/) we provide to experience it.

# Running on local
Download checkpoints for human parsing and dwpose from [here](https://huggingface.co/BoyuanJiang/FluxFit/tree/main) and put under checkpoints folder.

```
checkpoints
|-- dwpose
    |-- dw-ll_ucoco_384.onnx
    |-- yolox_l.onnx
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx    
```


You should first get access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and login your account with *huggingface-cli login*

Chinese users can refer to [this link](https://hf-mirror.com/) to speed up model downloads.

Run demo with following command, requiring a gpu that supports bf16.
```
pip install diffusers==0.30.3

python gradio_flux.py # need about 66G GPU memory

python gradio_flux.py --offload # need about 32G GPU memory, the speed is slower

python gradio_flux.py --aggressive_offload # need about 8G GPU memory, the speed is slowest

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BoyuanJiang/FluxFit&type=Date)](https://star-history.com/#BoyuanJiang/FluxFit&Date)

## License
The codes and checkpoints in this repository are under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Contact
The model is developed by [Boyuan Jiang](https://byjiang.com/) at Tencent Youtu Lab. If you are interested in collaboration, feel free to contact me.
