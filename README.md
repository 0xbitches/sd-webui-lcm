# Latent Consistency Model for Stable Diffusion WebUI <!-- omit from toc -->

![Extension Preview](./assets/preview.png)

This extension aims to integrate [Latent Consistency Model (LCM)](https://latent-consistency-models.github.io/) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Note that LCMs are a completely different class of models than Stable Diffusion, and the only available checkpoint currently is [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7).

**This is a very barebone implementation written in an hour, so any PRs are welcome.**

## Table of Contents <!-- omit from toc -->

- [Installation:](#installation)
- [Img2Img / Vid2Vid](#img2img--vid2vid)
- [Known Issues](#known-issues)
  - [1. `ValueError: Non-consecutive added token '<|startoftext|>' found. Should have index 49408 but has index 49406 in saved vocabulary.`](#1-valueerror-non-consecutive-added-token-startoftext-found-should-have-index-49408-but-has-index-49406-in-saved-vocabulary)
  - [2. `torch.cuda.OutOfMemoryError...`](#2-torchcudaoutofmemoryerror)

## Installation:

Simply clone this repo to your `extensions/` directory:

```
git clone https://github.com/0xbitches/sd-webui-lcm
```

Or go to "Extensions->Install from URL".

Then reload your WebUI.

Generated images will be saved to `outputs/txt2img-images/LCM`. You can use PNG Info to examine generation data.

## Img2Img / Vid2Vid

![vid2vid](./assets/vid2vid.png)

Img2Img/Vid2Vid with LCM is now supported in A1111. Simply update your extension and you should see the extra tabs.

Note that for these features, output height and width will be the same as input, and currently not changeable.

Generated videos will be saved to `outputs/LCM-vid2vid`.

## Known Issues

#### 1. `ValueError: Non-consecutive added token '<|startoftext|>' found. Should have index 49408 but has index 49406 in saved vocabulary.`

To resolve this, locate your huggingface hub cache directory.

It will be something like `~/.cache/huggingface/hub/path_to_lcm_dreamshaper_v7/tokenizer/`. On Windows, it will roughly be `C:\Users\YourUserName\.cache\huggingface\hub\models--SimianLuo--LCM_Dreamshaper_v7\snapshots\c7f9b672c65a664af57d1de926819fd79cb26eb8\tokenizer\`.

Find the file `added_tokens.json` and change the contents to:

```
{
  "<|endoftext|>": 49409,
  "<|startoftext|>": 49408
}
```

or simply remove it.

#### 2. `torch.cuda.OutOfMemoryError...`

This is because Automatic1111 loads an SD checkpoint on top of LCM.

Try Settings -> Actions -> Unload SD checkpoint to free VRAM
