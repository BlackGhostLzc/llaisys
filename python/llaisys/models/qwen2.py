from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import safetensors
import json
from ctypes import byref, c_int, c_size_t, c_float, c_int64, c_uint32, c_void_p

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    llaisysDataType_t,
    llaisysDeviceType_t,
)

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        model_path = Path(model_path)
        config_path = model_path / "config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        torch_dtype = str(cfg.get("torch_dtype", "bfloat16")).lower()
        if "float32" in torch_dtype or torch_dtype in {"fp32", "f32"}:
            dtype = DataType.F32
        elif "float16" in torch_dtype or torch_dtype in {"fp16", "f16"}:
            dtype = DataType.F16
        else:
            dtype = DataType.BF16
        # 统一用 torch 读取 bfloat16，并降级为 float16，避免 numpy bfloat16 兼容问题
        use_torch_loader = False
        if dtype == DataType.BF16:
            dtype = DataType.F16
            use_torch_loader = True
        # 解析模型参数
        nlayer = int(cfg.get("num_hidden_layers", 0))
        hs = int(cfg.get("hidden_size", 0))
        nh = int(cfg.get("num_attention_heads", 0))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg.get("intermediate_size", 0))
        maxseq = int(cfg.get("max_position_embeddings", 0))
        voc = int(cfg.get("vocab_size", 0))
        epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        eos = cfg.get("eos_token_id", -1)
        # 解析结束token
        if isinstance(eos, list):
            end_token = int(eos[0]) if eos else -1
        else:
            end_token = int(eos)
        # 解析head_dim
        dh = int(cfg.get("head_dim", hs // nh if nh else 0))
        

        # debug 信息
        print(f"\n{'='*20} Qwen2 Model Config Info {'='*20}")
        print(f"1.  Model Path:       {model_path}")
        print(f"2.  Compute Dtype:    {dtype} (Original: {torch_dtype})")
        print(f"3.  Use Torch Loader: {use_torch_loader}")
        print(f"4.  Layers (nlayer):  {nlayer}")
        print(f"5.  Hidden Size (hs): {hs}")
        print(f"6.  Attn Heads (nh):  {nh}")
        print(f"7.  KV Heads (nkvh):  {nkvh}  [{'GQA' if nkvh != nh else 'MHA'}]")
        print(f"8.  Head Dim (dh):    {dh}")
        print(f"9.  FFN Dim (di):     {di}")
        print(f"10. Vocab Size (voc): {voc}")
        print(f"11. Max Seq Len:      {maxseq}")
        print(f"12. RoPE Theta:       {theta}")
        print(f"13. Norm Epsilon:     {epsilon}")
        print(f"14. EOS Token ID:     {end_token}")
        print(f"{'='*60}\n")

        model_meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
            c_size_t(nlayer),
            c_size_t(hs),
            c_size_t(nh),
            c_size_t(nkvh),
            c_size_t(dh),
            c_size_t(di),
            c_size_t(maxseq),
            c_size_t(voc),
            c_float(epsilon),
            c_float(theta),
            c_int64(end_token),
        )

        print("python call llaisysQwen2ModelCreate")
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(model_meta),
            llaisysDeviceType_t(device),
            device_ids,
            1,
        )

        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")

            
        self._model_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._meta = model_meta



    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []
