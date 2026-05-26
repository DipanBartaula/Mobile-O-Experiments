from diffusers import AutoencoderDC, SanaTransformer2DModel
import torch


def build_sana(vision_tower_cfg, **kwargs):
    print(f"[checkpoint-load] dit: diffusion_name_or_path={vision_tower_cfg.diffusion_name_or_path}")
    print("="*20, "Building Sana Transformer", vision_tower_cfg.diffusion_name_or_path, "="*20)
    sana = SanaTransformer2DModel.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="transformer", low_cpu_mem_usage=False, ignore_mismatched_sizes=True, torch_dtype=torch.float16)
    return sana


def build_vae(vision_tower_cfg, **kwargs):
    print(f"[checkpoint-load] vae: diffusion_name_or_path={vision_tower_cfg.diffusion_name_or_path}")
    vae = AutoencoderDC.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="vae", torch_dtype=torch.float16)
    return vae

