"""DeepseekV4Processor — HF ProcessorMixin built on the OFFICIAL DeepSeek-V4
image_processor.py + encoding_dsv4.py logic (portable, no vllm dependency).

Text side:  encoding_dsv4.encode_messages  (chat rendering / apply_chat_template)
Image side: image_processor.load_image + build_image_block + prepare_vl_inputs

Official modules are loaded RELATIVE to this file's directory so the processor
package can be dropped into any HF model/processor directory as-is.
"""
from __future__ import annotations

import importlib.util as _ilu
import io
import os
import sys
import types
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, BatchFeature, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


IMAGE_SENTINEL_BASE_ID = 129257
IMAGE_SENTINEL_END_ID = IMAGE_SENTINEL_BASE_ID + 4          # 129261
IMAGE_TOKEN_ID = 129264                                      # fallback
_IMAGE_SENTINEL_CHECKED = False


def _validate_image_sentinel_ids(tokenizer) -> None:
    global _IMAGE_SENTINEL_CHECKED
    if _IMAGE_SENTINEL_CHECKED:
        return
    names = tokenizer.convert_ids_to_tokens([IMAGE_SENTINEL_BASE_ID + i for i in range(5)])
    for i, name in enumerate(names):
        if tokenizer.convert_tokens_to_ids(name) != IMAGE_SENTINEL_BASE_ID + i:
            raise ValueError(
                "DeepSeek-V4 vision sentinel block requires tokenizer ids "
                f"{IMAGE_SENTINEL_BASE_ID}..{IMAGE_SENTINEL_BASE_ID + 4}; found "
                f"{names!r}. Align IMAGE_SENTINEL_BASE_ID with "
                "vllm_ascend/models/deepseek_v4/mm_preprocess.py."
            )
    _IMAGE_SENTINEL_CHECKED = True


def _load_official(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_IMG = _load_official("image_processor", os.path.join(_BASE_DIR, "image_processor.py"))
_ENC = _load_official("encoding_dsv4", os.path.join(_BASE_DIR, "encoding_dsv4.py"))


def _resolve_image_token_id(tokenizer) -> int:
    """Return tokenizer's actual id for IMAGE_PLACEHOLDER, fallback IMAGE_TOKEN_ID."""
    if tokenizer is None:
        return IMAGE_TOKEN_ID
    try:
        tid = tokenizer.convert_tokens_to_ids(_ENC.IMAGE_PLACEHOLDER)
    except Exception:
        return IMAGE_TOKEN_ID
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return IMAGE_TOKEN_ID
    return int(tid)


def _normalize_image(img: Any) -> dict:
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return {"type": "image", "data": buf.getvalue()}
    if isinstance(img, (bytes, bytearray)):
        return {"type": "image", "data": bytes(img)}
    if isinstance(img, str):
        return {"type": "image", "url": img}
    if isinstance(img, dict):
        if "bytes" in img:
            return {"type": "image", "data": bytes(img["bytes"])}
        if "path" in img:
            return {"type": "image", "url": str(img["path"])}
        block = dict(img)
        block.pop("type", None)
        inner = block.get("image")
        if inner is not None and not any(k in block for k in ("source", "url", "data")):
            block.pop("image")
            return _normalize_image(inner)
        if any(k in block for k in ("source", "url", "data")):
            return {"type": "image", **{k: v for k, v in block.items() if k in ("source", "url", "data")}}
    raise TypeError(f"Cannot normalize image input: {type(img)}")


class DeepseekV4ImageProcessor(BaseImageProcessor):
    def __init__(self, patch_size=14, downsample_ratio=3, max_n_token=384,
                 min_pixels=147456, max_wh_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.downsample_ratio = int(downsample_ratio)
        self.max_n_token = int(max_n_token)
        self.min_pixels = int(min_pixels)
        self.max_wh_ratio = int(max_wh_ratio)

    def __call__(self, image: Any):
        args = types.SimpleNamespace(
            vision_patch_size=self.patch_size,
            vision_downsample_ratio=self.downsample_ratio,
            vision_max_n_token=self.max_n_token,
            vision_min_pixels=self.min_pixels,
            vision_max_wh_ratio=self.max_wh_ratio,
        )
        return _IMG.load_image(_normalize_image(image), args)


def _prepare_vl_inputs(prompt, images, tokenizer, args, image_token_id=None):
    if images:
        _validate_image_sentinel_ids(tokenizer)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids(_ENC.IMAGE_PLACEHOLDER)
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(1 for tok in prompt_tokens if tok == image_token_id)
    if num_placeholders != len(images):
        ph_tok = tokenizer.convert_ids_to_tokens(image_token_id)
        ph_special = image_token_id in getattr(tokenizer, "all_special_ids", [])
        const_special = IMAGE_TOKEN_ID in getattr(tokenizer, "all_special_ids", [])
        raise ValueError(
            f"Found {num_placeholders} image tokens but got {len(images)} images. "
            f"image_token_id={image_token_id} (token={ph_tok!r}, is_special={ph_special}); "
            f"IMAGE_TOKEN_ID={IMAGE_TOKEN_ID} is_special={const_special}; "
            f"sentinel range=({IMAGE_SENTINEL_BASE_ID}..{IMAGE_SENTINEL_END_ID}); "
            f"prompt head={ (prompt[:200] if isinstance(prompt, str) else repr(prompt)[:200])!r }. "
            "Hints: (A) agent_loop.py::_compute_multi_modal_inputs must call "
            "processor.collapse_media_blocks(input_ids) BEFORE "
            "decode(skip_special_tokens=True); (B) image_token_id must be a "
            "NON-special in-vocab token so it survives decode+re-encode."
        )
    tokens, image_inputs = [], []
    it = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = _IMG.load_image(next(it), args)
        types, perm = _IMG.build_image_block(n_llm_h, n_llm_w, len(tokens))
        image_inputs.append({
            "start": len(tokens),
            "patches": patches,
            "n_vit_h": n_vit_h, "n_vit_w": n_vit_w,
            "n_llm_h": n_llm_h, "n_llm_w": n_llm_w,
            "types": types, "perm": perm,
        })
        tokens += (IMAGE_SENTINEL_BASE_ID + types).tolist()
    if not image_inputs:
        return tokens, None
    return tokens, image_inputs


class DeepseekV4Processor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]

    def __init__(self, tokenizer=None, image_processor=None, vocab_size=129280,
                 chat_template=None, **kwargs):
        if image_processor is None or not hasattr(image_processor, "patch_size"):
            image_processor = DeepseekV4ImageProcessor()
        from transformers.image_processing_utils import BaseImageProcessor, ImageProcessingMixin
        kwargs.setdefault("tokenizer", tokenizer)
        kwargs.setdefault(
            "image_processor",
            image_processor if isinstance(image_processor, ImageProcessingMixin) else BaseImageProcessor(),
        )
        super().__init__(**kwargs)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.vocab_size = int(vocab_size)

        # 启动自检：占位符 id 必须是 in-vocab 且非 special。
        self._resolved_image_token_id = _resolve_image_token_id(tokenizer)
        if tokenizer is not None:
            if self._resolved_image_token_id != IMAGE_TOKEN_ID:
                warnings.warn(
                    f"IMAGE_TOKEN_ID constant ({IMAGE_TOKEN_ID}) != tokenizer id for "
                    f"{_ENC.IMAGE_PLACEHOLDER!r} ({self._resolved_image_token_id}); "
                    "collapse_media_blocks will use the tokenizer's id.",
                    RuntimeWarning,
                )
            if self._resolved_image_token_id in getattr(tokenizer, "all_special_ids", []):
                warnings.warn(
                    f"image placeholder id {self._resolved_image_token_id} is a SPECIAL "
                    "token — decode(skip_special_tokens=True) will drop it and the "
                    "agent_loop round-trip cannot work. Choose a non-special in-vocab "
                    "placeholder id.",
                    RuntimeWarning,
                )

    def save_pretrained(self, save_directory, **kwargs):
        # Preserve custom source and all image parameters for an offline reload.
        import json
        import shutil
        from pathlib import Path

        files = super().save_pretrained(save_directory, **kwargs)
        target = Path(save_directory)
        config = {
            "vocab_size": self.vocab_size,
            "image_processor": self.image_processor.to_dict(),
            "chat_template": self.chat_template,
        }
        (target / "dsv4_processor_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for name in ("processing_deepseek_v4.py", "image_processor.py", "encoding_dsv4.py"):
            source = Path(_BASE_DIR) / name
            if source.resolve() != (target / name).resolve():
                shutil.copy2(source, target / name)
        return files

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        config_path = Path(pretrained_model_name_or_path) / "dsv4_processor_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        vocab_size = kwargs.pop("vocab_size", config.get("vocab_size", 129280))
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer=tokenizer,
                   image_processor=DeepseekV4ImageProcessor(**config.get("image_processor", {})),
                   vocab_size=vocab_size, chat_template=config.get("chat_template"))

    @property
    def image_token_id(self) -> int:
        return self._resolved_image_token_id

    def collapse_media_blocks(self, token_ids):
        """Collapse each expanded [IMAGE_START..IMAGE_END] span to a single placeholder."""
        if token_ids is None:
            return token_ids
        ids = list(token_ids)
        if not ids:
            return ids
        out: List[int] = []
        i, n = 0, len(ids)
        start_id = IMAGE_SENTINEL_BASE_ID
        end_id = IMAGE_SENTINEL_END_ID
        placeholder_id = self.image_token_id
        while i < n:
            tok = ids[i]
            if tok != start_id:
                out.append(tok); i += 1; continue
            j = i + 1
            while j < n and ids[j] != end_id:
                j += 1
            if j >= n:
                out.append(tok); i += 1; continue
            out.append(placeholder_id)
            i = j + 1
        return out

    @property
    def _vision_args(self):
        ip = self.image_processor
        return types.SimpleNamespace(
            vision_patch_size=ip.patch_size,
            vision_downsample_ratio=ip.downsample_ratio,
            vision_max_n_token=ip.max_n_token,
            vision_min_pixels=ip.min_pixels,
            vision_max_wh_ratio=ip.max_wh_ratio,
            vocab_size=self.vocab_size,
        )

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kwargs):
        thinking_mode = kwargs.pop("thinking_mode", "chat")
        drop_thinking = kwargs.pop("drop_thinking", True)
        add_default_bos_token = kwargs.pop("add_default_bos_token", True)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        tools = kwargs.pop("tools", None)

        norm: List[Dict[str, Any]] = []
        for msg in messages:
            m = dict(msg)
            if tools is not None and m.get("role") == "user":
                m["tools"] = tools
            content = m.get("content")
            if isinstance(content, str):
                norm.append(m); continue
            blocks: List[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    blocks.append(_normalize_image(item))
                elif isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"type": "text", "text": str(item.get("text", ""))})
                elif isinstance(item, dict):
                    blocks.append(item)
                elif isinstance(item, str):
                    blocks.append({"type": "text", "text": item})
            m["content"] = blocks
            norm.append(m)

        prompt = _ENC.encode_messages(
            norm,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
        )
        if add_generation_prompt:
            prompt = prompt.rstrip() + _ENC.ASSISTANT_SP_TOKEN
        if tokenize:
            return self.tokenizer.encode(prompt)
        return prompt

    def __call__(self, text=None, images=None, videos=None, audio=None,
                 return_tensors=None, **kwargs):
        del videos, audio, kwargs
        texts = [text] if isinstance(text, str) else list(text or [])
        records = [_normalize_image(img) for img in (images or [])]

        all_input_ids, all_attention_mask = [], []
        all_patches, all_vit_grid, all_llm_grid, all_perm, all_types = [], [], [], [], []
        for t in texts:
            tokens, image_inputs = _prepare_vl_inputs(
                t, records, self.tokenizer, self._vision_args,
                image_token_id=self.image_token_id,
            )
            all_input_ids.append(tokens)
            all_attention_mask.append([1] * len(tokens))
            for ii in image_inputs or []:
                all_patches.append(ii["patches"])
                all_vit_grid.append((ii["n_vit_h"], ii["n_vit_w"]))
                all_llm_grid.append((ii["n_llm_h"], ii["n_llm_w"]))
                all_perm.append(ii["perm"])
                all_types.append(ii["types"])

        if return_tensors == "pt":
            out = {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long),
            }
            if all_patches:
                out.update({
                    "patches": torch.cat(all_patches),
                    "vit_grid": torch.tensor(all_vit_grid, dtype=torch.int64),
                    "llm_grid": torch.tensor(all_llm_grid, dtype=torch.int64),
                    "perm": torch.cat(all_perm),
                    "types": torch.cat(all_types),
                })
            return BatchFeature(out)
        out = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}
        if all_patches:
            out.update({
                "patches": torch.cat(all_patches),
                "vit_grid": all_vit_grid,
                "llm_grid": all_llm_grid,
                "perm": torch.cat(all_perm).tolist(),
                "types": torch.cat(all_types).tolist(),
            })
        return out