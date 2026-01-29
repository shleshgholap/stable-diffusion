"""CLIP Text Encoder for Stable Diffusion."""

from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        freeze: bool = True,
        layer: int = -1,
        use_attention_mask: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.layer = layer
        self.use_attention_mask = use_attention_mask
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        
        if freeze:
            self.freeze()
        
        self.hidden_size = self.text_model.config.hidden_size
    
    def freeze(self) -> None:
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
    
    def unfreeze(self) -> None:
        for param in self.text_model.parameters():
            param.requires_grad = True
    
    @property
    def device(self) -> torch.device:
        return next(self.text_model.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.text_model.parameters()).dtype
    
    def tokenize(self, texts: Union[str, List[str]], padding: str = "max_length", truncation: bool = True) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, padding=padding, max_length=self.max_length, truncation=truncation, return_tensors="pt")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask if self.use_attention_mask else None,
            output_hidden_states=True,
        )
        if self.layer == -1:
            return outputs.last_hidden_state
        return outputs.hidden_states[self.layer]
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        batch_encoding = self.tokenize(texts)
        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)
        return self.forward(input_ids, attention_mask)
    
    def get_unconditional_embeddings(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        uncond_tokens = self.tokenize([""] * batch_size)
        uncond_input_ids = uncond_tokens["input_ids"].to(device)
        uncond_attention_mask = uncond_tokens["attention_mask"].to(device)
        return self.forward(uncond_input_ids, uncond_attention_mask)
    
    def encode_with_pooled(self, texts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_encoding = self.tokenize(texts)
        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)
        
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask if self.use_attention_mask else None,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.last_hidden_state if self.layer == -1 else outputs.hidden_states[self.layer]
        return hidden_states, outputs.pooler_output


class FrozenCLIPTextEncoder(CLIPTextEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", max_length: int = 77, layer: int = -1):
        super().__init__(model_name=model_name, max_length=max_length, freeze=True, layer=layer)
    
    def train(self, mode: bool = True):
        return super().train(False)


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder: CLIPTextEncoder, cache_embeddings: bool = False):
        super().__init__()
        self.text_encoder = text_encoder
        self.cache_embeddings = cache_embeddings
        self._cache = {}
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if not self.cache_embeddings:
            return self.text_encoder.encode(texts)
        
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            embeddings = self.text_encoder.encode(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                self._cache[text] = emb.clone()
                results.append((idx, emb))
        
        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results])
    
    def clear_cache(self):
        self._cache = {}
