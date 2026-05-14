import torch
import math
import transformers
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import torch.nn as nn

OLD_LLAMA_FORWARD = transformers.models.mistral.modeling_mistral.MistralAttention.forward

def med_opera_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        
        bsz, q_len, _ = hidden_states.size()

        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) #按“头”拆分，并把“头”摆到正确的位置，好让每个“头”独立去计算注意力
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # print("[*] 进入 Med-OPERA Attention Forward，正在处理输入张量...")

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)#value向量不进行旋转

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            #torch.cat封装在了Cache对象的update方法,返回的是全新的key_states和value_states，包含了当前的key/value和之前缓存的key/value
        
            

        key_states = repeat_kv(key_states, self.num_key_value_groups) #复制key和value向量，使得它们的“头”数量和query的“头”数量一致
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        
       
        if hasattr(self, "med_opera_boost_factor") and self.med_opera_boost_factor>0:
            
            
            boost_factor = getattr(self, "med_opera_boost_factor")
            global_targets = getattr(self,"med_opera_target_patches",None)
            region_prior = getattr(self,"region_prior",None )
            max_len = attn_weights.shape[-1]

            valid_mask = (global_targets >= 0) & (global_targets < max_len)
            valid_targets = global_targets[valid_mask]
            valid_prior = region_prior[valid_mask].to(
                device=attn_weights.device,
                dtype=attn_weights.dtype
            )

            valid_prior = valid_prior.view(1, 1, 1, -1)

            
            if valid_targets.numel() > 0:    
                # print("ok, applying Med-OPERA attention boost on targets:", valid_targets, "with region_prior:", valid_prior, "and boost_factor:", boost_factor)
                target_weights = attn_weights[:, :, :, valid_targets]
                attn_weights[:, :, :, valid_targets] = target_weights + target_weights.abs() * valid_prior * boost_factor
                
                
        
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        
        if attention_mask is not None:  # no matter the length, we just slice it
            
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # print(causal_mask)
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        # print(f"[*] Attention weights after Med-OPERA intervention (first head, first query): {attn_weights[0, 0, 0, :50]}")
        #先将数据转成 float32 进行 Softmax，得到高精度的概率分布后，再转回模型原始的低精度格式（query_states.dtype）
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) 
        # print(f"[*] Attention weights after dropout (first head, first query): {attn_weights[0, 0, 0, :50]}")
        #在推理（Inference）时，self.training 为 False，这一行相当于什么都不做
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            
            print(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            

        attn_output = attn_output.transpose(1, 2).contiguous() #在显存中开辟一块全新的连续空间来存储转置后的张量

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        # print(f"here{self.layer_idx}")
        return attn_output, attn_weights, past_key_value

def apply_med_opera_patch(model, start_layer=0, end_layer=None):
    """
    终极动态挂载函数（指定层干预版）：
    使用 types.MethodType 仅对[start_layer, end_layer) 区间内的 Attention 实例进行独立覆写。
    """
    import types
    
    # 剥开 LLaVA 的壳，拿到包含所有 layers 的基础模型
    base_model = getattr(model, "model", model) 
    
    # 如果不指定 end_layer，默认干预到最后一层
    total_layers = len(base_model.layers)
    if end_layer is None:
        end_layer = total_layers
        
    # 安全检查，防止索引越界
    start_layer = max(0, start_layer)
    end_layer = min(total_layers, end_layer)
    
    print(f"[*] 🚀 准备进行 Med-OPERA 局部网络干预 (目标：Layer {start_layer} 到 Layer {end_layer-1})")
    
    for i in range(start_layer, end_layer):
        # 1. 拿到具体某一层的 self_attn 实例对象
        attn_instance = base_model.layers[i].self_attn
        
        attn_instance.med_opera_target_patches = None
        attn_instance.med_opera_boost_factor = 0.0
        attn_instance.use_cfg = False
        
        attn_instance.forward = types.MethodType(med_opera_attention_forward, attn_instance)
        
    print(f"[*] ✅ 挂载成功！已对指定的 {end_layer - start_layer} 层 Attention 完成了外科手术式替换。")
    print(f"[*] 未指定的层将继续使用原生 Attention 引擎。")