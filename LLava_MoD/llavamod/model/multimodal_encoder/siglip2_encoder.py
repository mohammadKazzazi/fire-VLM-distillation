import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class Siglip2VisionTower(nn.Module):
    """SigLIP2 vision tower (e.g. google/siglip2-base-patch16-384).

    Loads via AutoModel, keeps vision_model only (~90M params), drops text tower.
    """

    def __init__(self, image_tower, args, delay_load=False, cache_dir="./cache_dir"):
        super().__init__()

        self.is_loaded = False
        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        local_files_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        processor = AutoProcessor.from_pretrained(
            self.image_tower_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        full_model = AutoModel.from_pretrained(
            self.image_tower_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )

        self.image_processor = processor.image_processor
        self.image_tower = full_model.vision_model
        self.image_tower.requires_grad_(False)

        # Release SigLIP text tower — LLaVA uses the LM for text, not SigLIP text encoder.
        for attr in ("text_model", "text_projection", "logit_scale", "logit_bias"):
            if hasattr(full_model, attr):
                delattr(full_model, attr)
        del full_model

        self._hidden_size = int(self.image_tower.config.hidden_size)
        self._image_size = getattr(self.image_tower.config, "image_size", 384)
        if isinstance(self._image_size, (list, tuple)):
            self._image_size = self._image_size[0]
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # SigLIP2 vision forward does not expose hidden_states in transformers 4.57;
        # use last_hidden_state (equivalent to final layer patches).
        image_features = image_forward_outs.last_hidden_state
        if self.select_feature == "patch":
            return image_features
        if self.select_feature == "cls_patch":
            cls = image_features.mean(dim=1, keepdim=True)
            return torch.cat([cls, image_features], dim=1)
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()

        if isinstance(images, list):
            feats = []
            for image in images:
                out = self.image_tower(
                    pixel_values=image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    return_dict=True,
                )
                feats.append(self.feature_select(out).to(image.dtype))
            return feats

        out = self.image_tower(
            pixel_values=images.to(device=self.device, dtype=self.dtype),
            return_dict=True,
        )
        return self.feature_select(out).to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if not self.is_loaded:
            return torch.float32
        return next(self.image_tower.parameters()).dtype

    @property
    def device(self):
        if not self.is_loaded:
            return torch.device("cpu")
        return next(self.image_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        class _Cfg:
            hidden_size = 768
            image_size = 384
            patch_size = 16

        return _Cfg()

    @property
    def hidden_size(self):
        return self._hidden_size if self.is_loaded else 768

    @property
    def num_patches(self):
        if not self.is_loaded:
            return 576
        size = self._image_size
        patch = getattr(self.image_tower.config, "patch_size", 16)
        return (size // patch) ** 2
