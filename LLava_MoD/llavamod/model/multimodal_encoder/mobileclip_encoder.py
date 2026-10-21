"""MobileCLIP vision tower via open_clip (patch features for LLaVA-MoD)."""
import os
from typing import Callable, List, Union

import torch
import torch.nn as nn
from PIL import Image

try:
    import open_clip
except ImportError as e:
    open_clip = None
    _OPEN_CLIP_IMPORT_ERROR = e
else:
    _OPEN_CLIP_IMPORT_ERROR = None


class MobileCLIPImageProcessor:
    """Minimal processor API compatible with llavamod/mm_utils.process_images."""

    def __init__(self, transform: Callable, image_mean=(0.0, 0.0, 0.0)):
        self.transform = transform
        self.image_mean = image_mean

    def preprocess(self, image: Image.Image, return_tensors="pt"):
        tensor = self.transform(image.convert("RGB"))
        if return_tensors == "pt" and tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return {"pixel_values": tensor}


def _parse_mobileclip_spec(image_tower: str):
    # Formats:
    #   mobileclip:MobileCLIP-S2
    #   mobileclip:MobileCLIP-S2:datacompdr
    #   open_clip/MobileCLIP-S2
    spec = image_tower
    if spec.startswith("mobileclip:"):
        spec = spec[len("mobileclip:"):]
    elif spec.startswith("open_clip/"):
        spec = spec[len("open_clip/"):]
    parts = spec.split(":")
    model_name = parts[0]
    pretrained = parts[1] if len(parts) > 1 else "datacompdr"
    return model_name, pretrained


class MobileCLIPVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir="./cache_dir"):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "MobileCLIP requires open-clip-torch. "
                "Install: pip install open-clip-torch"
            ) from _OPEN_CLIP_IMPORT_ERROR

        self.is_loaded = False
        self.image_tower_name = image_tower
        self.model_name, self.pretrained_tag = _parse_mobileclip_spec(image_tower)
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        local_files_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        if local_files_only:
            raise RuntimeError(
                "MobileCLIP offline load is not configured. "
                "Pre-download open_clip weights or unset TRANSFORMERS_OFFLINE."
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained_tag
        )
        self.openclip_model = model
        self.visual_trunk = model.visual.trunk
        self.visual_trunk.requires_grad_(False)
        self.openclip_model.eval()

        self.image_processor = MobileCLIPImageProcessor(preprocess)
        self._hidden_size = int(getattr(self.visual_trunk, "num_features", 1280))
        self._image_size = getattr(model.visual, "image_size", (256, 256))
        if isinstance(self._image_size, int):
            self._image_size = (self._image_size, self._image_size)
        self.is_loaded = True

    def _spatial_to_patches(self, feat: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C]
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D spatial features, got {feat.shape}")
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2).contiguous()

    def feature_select(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        patches = self._spatial_to_patches(spatial_feat)
        if self.select_feature == "patch":
            return patches
        if self.select_feature == "cls_patch":
            # No CLS token; prepend mean pool as pseudo-cls
            cls = patches.mean(dim=1, keepdim=True)
            return torch.cat([cls, patches], dim=1)
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()

        if isinstance(images, list):
            feats = []
            for image in images:
                out = self.visual_trunk.forward_features(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                feats.append(self.feature_select(out).to(image.dtype))
            return feats

        out = self.visual_trunk.forward_features(
            images.to(device=self.device, dtype=self.dtype)
        )
        return self.feature_select(out).to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if not self.is_loaded:
            return torch.float32
        return next(self.visual_trunk.parameters()).dtype

    @property
    def device(self):
        if not self.is_loaded:
            return torch.device("cpu")
        return next(self.visual_trunk.parameters()).device

    @property
    def config(self):
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.hidden_size = self.hidden_size
        cfg.image_size = self._image_size[0] if self.is_loaded else 256
        cfg.patch_size = 16
        return cfg

    @property
    def hidden_size(self):
        if self.is_loaded:
            return self._hidden_size
        return 1280

    @property
    def num_patches(self):
        if not self.is_loaded:
            return 64
        # infer from a dry shape: FastViT on 256 -> 8x8
        size = self._image_size[0]
        return (size // 32) ** 2
