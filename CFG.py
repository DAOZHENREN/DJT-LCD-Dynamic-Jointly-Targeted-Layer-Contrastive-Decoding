
"""
External Med-OPERA generation patch.

Usage:
    from med_opera_generation import patch_med_opera
    patch_med_opera(model)
    out = model.generate(..., med_opera_decoding=True, num_beams=5)

This file is designed to keep Med-OPERA code out of transformers/generation/utils.py.
"""

import copy
import inspect
import types
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from transformers.generation.utils import (
        GenerationMixin,
        GenerationMode,
        validate_stopping_criteria,
        GenerateOutput,
        GenerateNonBeamOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        GenerateEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
    )
except Exception:  # pragma: no cover - for older/newer transformers layouts
    from transformers.generation.configuration_utils import GenerationMode
    from transformers.generation.utils import GenerationMixin, validate_stopping_criteria
    from transformers.generation import (
        GenerateOutput,
        GenerateNonBeamOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
    )

try:
    from transformers.generation import (
        BeamScorer,
        BeamSearchScorer,
        LogitsProcessorList,
        StoppingCriteriaList,
        GenerationConfig,
    )
except Exception:  # pragma: no cover - for older transformers layouts
    from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    from transformers.generation.configuration_utils import GenerationConfig

try:
    from transformers.deepspeed import is_deepspeed_zero3_enabled
except Exception:  # pragma: no cover
    def is_deepspeed_zero3_enabled() -> bool:
        return False

try:
    from transformers.utils import logging
    logger = logging.get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


_MED_OPERA_INTERNAL_KWARGS = {"cd_alpha", "cd_beta"}

# Original HF GenerationMixin.generate saved by install_med_opera_generation_patch().
# Keep this at module scope so LLaVA/LLaVA-Med top-level generate() can stay untouched
# while its super().generate(...) call enters generate_with_med_opera().
_ORIGINAL_HF_GENERATE = None


def _dist_world_size_gt_one() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def scale_embeddings_by_mask(inputs_embeds, mask, alpha=0.01):
    """
    inputs_embeds: [B, T, D]
    mask:          [B, T], True 表示该 token embedding 要被放缩
    """
    emb = inputs_embeds.clone()

    mask = mask.to(device=emb.device, dtype=torch.bool)
    scale = torch.ones(emb.shape[:2], device=emb.device, dtype=emb.dtype)
    scale.masked_fill_(mask, alpha)

    return emb * scale.unsqueeze(-1)

def _call_original_hf_generate(
    self,
    inputs=None,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    negative_prompt_ids=None,
    negative_prompt_attention_mask=None,
    **kwargs,
):
    """Call the unmodified HF GenerationMixin.generate."""
    original_generate = getattr(generate_with_med_opera, "_original_hf_generate", None)
    if original_generate is None:
        original_generate = _ORIGINAL_HF_GENERATE

    # Backward compatibility with the old instance-local patch.    if original_generate is None and hasattr(self, "_hf_generate_original"):
        return self._hf_generate_original(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

    if original_generate is None:
        raise RuntimeError(
            "Original HF generate is not available. "
            "Call install_med_opera_generation_patch() or patch_med_opera(model) first."
        )

    return original_generate(
        self,
        inputs=inputs,
        generation_config=generation_config,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        synced_gpus=synced_gpus,
        assistant_model=assistant_model,
        streamer=streamer,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        **kwargs,
    )


def install_med_opera_generation_patch():
    """Patch transformers GenerationMixin.generate, not model.generate.

    This is the LLaVA/LLaVA-Med-safe path: the model's own generate() remains
    responsible for multimodal preprocessing. Only its internal super().generate(...)
    call is routed through generate_with_med_opera().
    """
    global _ORIGINAL_HF_GENERATE

    if getattr(GenerationMixin.generate, "_med_opera_patched", False):
        return

    _ORIGINAL_HF_GENERATE = GenerationMixin.generate
    generate_with_med_opera._original_hf_generate = _ORIGINAL_HF_GENERATE
    generate_with_med_opera._med_opera_patched = True
    GenerationMixin.generate = generate_with_med_opera


def uninstall_med_opera_generation_patch():
    """Restore the original HF GenerationMixin.generate."""
    global _ORIGINAL_HF_GENERATE

    if _ORIGINAL_HF_GENERATE is not None:
        GenerationMixin.generate = _ORIGINAL_HF_GENERATE
        _ORIGINAL_HF_GENERATE = None


def generate_with_med_opera(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    
    cd_mode: Optional[str] = None,
    jointed_cd_layer: Optional[int] = 8,
    key_position: Optional[dict] = None,
    scale_factor: float = 50.0,
    boost_factor: float = 0.5,
    penalty_weights: float = 1.0,
    **kwargs,
):
    """Drop-in generate wrapper.

    Non-Med-OPERA calls are delegated to the original HF `generate`.
    Med-OPERA calls execute only the beam-search path and call `self.med_opera_beam_search`.
    """
    # print("generate_with_med_opera called with med_opera_decoding =", med_opera_decoding)
    if cd_mode == None:
        return _call_original_hf_generate(
            self,
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

    if not hasattr(self, "med_opera_beam_search") and not hasattr(self, "djt_lcd_greedy_search"):
        raise RuntimeError(
            "med_opera_decoding=True requires patch_med_opera(model) on this model instance "
            "so med_opera_beam_search or djt_lcd_greedy_search is attached."
        )

    if streamer is not None:
        raise ValueError("Med-OPERA beam search does not support streamer. Set streamer=None.")

    if synced_gpus is None:
        synced_gpus = bool(is_deepspeed_zero3_enabled() and _dist_world_size_gt_one())

    self._validate_model_class()

    if generation_config is None:
        if (
            self.generation_config._from_model_config
            and self.generation_config._original_object_hash == hash(self.generation_config)
            and self.config._has_non_default_generation_parameters()
        ):
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. "
                    "Please prefer modifying model.generation_config instead.",
                    UserWarning,
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    # Med-OPERA consumes decoder self-attentions, so force this in the patched path.
    # generation_config.output_attentions = True
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = self._get_generation_mode(generation_config, assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    prepared_stopping_criteria = stopping_criteria

    # 10. basic checks shared by greedy / beam
    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")

    # ------------------------------------------------------------
    # 11. greedy search entry
    # ------------------------------------------------------------
    # 真正的 greedy decode 条件：
    #   num_beams == 1
    #   do_sample == False
    #
    # 注意：这里不 expand input_ids，也不创建 BeamSearchScorer。
    if generation_config.num_beams == 1 and generation_config.do_sample is False:
        if not hasattr(self, "djt_lcd_greedy_search"):
            raise RuntimeError(
                "generation_config.num_beams=1 requires self.djt_lcd_greedy_search. "
                "Please attach djt_lcd_greedy_search in patch_med_opera(model)."
            )

        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "`num_return_sequences` has to be 1 when using greedy search."
            )

        if cd_mode != "djt_lcd":
            raise ValueError(
                "djt_lcd_greedy_search only supports cd_mode='djt_lcd'. "
                f"Got cd_mode={cd_mode}."
            )

        return self.djt_lcd_greedy_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer = streamer,

            jointed_cd_layer=jointed_cd_layer,
            boost_factor=boost_factor,
            key_position=key_position,
            scale_factor=scale_factor,
            penalty_weights=penalty_weights,
            **model_kwargs,
        )

    # 11. prepare beam search scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=generation_config.num_beams,
        device=inputs_tensor.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        max_length=generation_config.max_length,
    )
    #管理每个 batch 的 beam hypotheses. .process() 选下一步 beam .finalize() 输出最终序列 
    # OPERA 并没有改这个模块，而是改了“每一步 token score 怎么算”。

    # 12. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_beams,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )
    #扩展后变成 [B*num_beams, T]
    assert generation_config.output_attentions, "OPERA decoding requires output_attentions=True!"   
    #OPERA 的核心就是看 self-attention 来做惩罚/回滚（rollback），所以必须拿到 attentions。

    

    if generation_config.num_return_sequences > generation_config.num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
    

    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    
    # 13. run opera beam search
    return self.med_opera_beam_search(
        input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=synced_gpus,
        
        cd_mode = cd_mode,
        jointed_cd_layer = jointed_cd_layer,
        boost_factor=boost_factor,
        penalty_weights=penalty_weights,
        key_position=key_position,
        scale_factor=scale_factor,
        **model_kwargs,
    )

def med_opera_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        #自定义变量
        cd_mode: Optional[str] = "djt_lcd",
        jointed_cd_layer: Optional[int] = 8,
        key_position: Optional[dict] = None,
        scale_factor: Optional[float] = 50.0,
        
        boost_factor: Optional[float] = 0.5,
        window_size: Optional[int] = 512, 
        penalty_weights: Optional[float] = 1.0,

        **model_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)

        num_beams = beam_scorer.num_beams
        # print(f"batch_size:{batch_size}\n")
        # print(f"num_beams:{num_beams}")
        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        # initialise the history variables
        
        beam_next_tokens = None
        beam_idx = None
        
        
        if cd_mode == "early_exit_cd" or cd_mode == "full_cd":
            cd_early_exit_layer = None
            model_kwargs_cd = model_kwargs.copy()
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            # Define current states
            current_state = {}
            current_state["input_ids"] = input_ids.clone()
            current_state["beam_scorer"] = copy.deepcopy(beam_scorer)
            current_state["beam_indices"] = beam_indices.copy() if beam_indices is not None else None  #保存每一步每条 beam 来自哪条父 beam
            current_state["cur_len"] = cur_len
            
            base_model = get_base_model(self)
            for layer in base_model.layers:
                layer.self_attn.med_opera_boost_factor = boost_factor 
            
            
            
            
            need_hidden_states = (
                cd_mode == "djt_lcd"
                or (cd_mode == "early_exit_cd" and cd_early_exit_layer is None)
            )
            # prepare model inputs 
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=need_hidden_states,
            )
            
            next_token_logits = outputs.logits[:, -1, :] # [batch_size * num_beams,vocab_size]   -1 切片不保留长度为1的维度
            for layer in base_model.layers:
                    layer.self_attn.med_opera_boost_factor = 0.0
            
            if cd_mode == "early_exit_cd" or cd_mode == "full_cd":
                model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

            
            if cd_mode == "full_cd":
                # 原始 CD：完整 forward
                outputs_cd = self(
                    **model_inputs_cd,
                    return_dict=True,
                )
            elif cd_mode == "early_exit_cd":
                # 更简单的 early-exit CD：
                # 临时只保留前 N 层，然后仍然调用原来的 self(...)
                if cd_early_exit_layer is None:
                    candidate_cd_early_exit_layers = [1, 2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16, 20, 24,28,31]
                    cd_early_exit_layer, cd_layer_scores = select_dola_like_early_exit_layer(
                        model=self,
                        outputs=outputs,
                        candidate_early_exit_layers=candidate_cd_early_exit_layers,
                        )
                    print(cd_early_exit_layer, cd_layer_scores)
                if not hasattr(base_model, "layers") and hasattr(base_model, "model"):
                    base_model = base_model.model

                original_layers = base_model.layers

                try:
                    base_model.layers = nn.ModuleList(
                        list(original_layers[:cd_early_exit_layer])
                    )

                    outputs_cd = self(
                        **model_inputs_cd,
                        return_dict=True,
                    )
                    
                finally:
                    # 必须恢复完整层数
                    base_model.layers = original_layers
            elif cd_mode == "djt_lcd":
                outputs_cd = None
                next_token_logits_cd = None
                # print(cd_mode)
                if outputs.hidden_states is None:
                    raise ValueError(
                        "djt_lcd mode requires output_hidden_states=True."
                    )
                

                # mature layer logits，也就是最终层 logits
                final_logits = outputs.logits[:, -1, :]

                # premature layer logits，也就是被选中的中间层 logits
                premature_logits = hidden_to_logits(
                    model=self,
                    base_model=base_model,
                    hidden_states=outputs.hidden_states,
                    exit_layer=jointed_cd_layer,
                )

                # DoLa-style relative top filtering
                # 如果你类里已经有 relative_top_filter，就直接用 self.relative_top_filter
                relative_top=0.1
                if relative_top > 0.0:
                    final_logits = self.relative_top_filter(
                        final_logits,
                        relative_top=relative_top,
                    )

                    premature_logits = premature_logits.log_softmax(dim=-1)

                    # final_logits 被过滤掉的位置，premature_logits 也同步压低
                    mask = final_logits < -1e3
                    premature_logits = premature_logits.masked_fill(mask, -1e3)

                # 纯 DoLa 核心：
                # next_token_logits 直接变成 final - premature
                next_token_logits = final_logits - premature_logits
            else:
                raise ValueError(f"Unknown cd_mode: {cd_mode}")
            
            if cd_mode == "early_exit_cd" or cd_mode == "full_cd":
                next_token_logits_cd = outputs_cd.logits[:, -1, :]
                ## cd_comments: pre-process logits from contrastive inputs
                cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
                
                cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                
                diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
                next_token_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) #这是 HF 通用兼容逻辑，一般不用太管（主要防某些模型生成 pad token 的问题）
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)  #在“模型概率”基础上再加规则
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores) #加上历史 beam 累计分数（形成扩展总分）

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor") #div除法
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]


            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
            
            if cd_mode == "early_exit_cd" or cd_mode == "full_cd":
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if "past_key_values" in model_kwargs_cd and model_kwargs_cd["past_key_values"] is not None:
                    model_kwargs_cd["past_key_values"] = self._reorder_cache(model_kwargs_cd["past_key_values"], beam_idx)
            
            
            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )
        # print("--------------------------------------\n")
        # print(sequence_outputs["sequences"].shape)
        #输出结果为：torch.Size([1, 26]) 说明返回了一束作为最终模型输出
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                ) #这个函数把 beam search 的结果打包成 HuggingFace 的标准“生成输出对象”
        else:
            return sequence_outputs["sequences"]
        
def relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized

def select_dola_like_early_exit_layer(
    model,
    outputs,
    candidate_early_exit_layers,
):
    """
    使用 DoLa 风格的 JS divergence 选择 early-exit 层。

    candidate_early_exit_layers: 1-based list
    例如 [1, 2, 4, 8, 12]
    """

    if outputs.hidden_states is None:
        raise ValueError("outputs.hidden_states is None. You must set output_hidden_states=True.")

    base_model = get_base_model(model)
    num_layers = len(base_model.layers)

    candidate_early_exit_layers = [
        l for l in candidate_early_exit_layers
        if 1 <= l < num_layers
    ]

    if len(candidate_early_exit_layers) == 0:
        raise ValueError("No valid candidate early-exit layers.")

    hidden_states = outputs.hidden_states

    # 最终成熟层 logits
    mature_logits = outputs.logits[:, -1, :].float()

    # 候选 early-exit 层 logits
    premature_logits = torch.stack(
        [
            hidden_to_logits(
                model=model,
                base_model=base_model,
                hidden_states=hidden_states,
                exit_layer=l,
            ).float()
            for l in candidate_early_exit_layers
        ],
        dim=0,
    )

    # mature distribution
    p_mature = F.softmax(mature_logits, dim=-1)

    # premature distributions
    p_premature = F.softmax(premature_logits, dim=-1)

    # mixture distribution
    m = 0.5 * (p_mature[None, :, :] + p_premature)

    log_p_mature = F.log_softmax(mature_logits, dim=-1)[None, :, :]
    log_p_premature = F.log_softmax(premature_logits, dim=-1)

    # JS divergence
    kl1 = F.kl_div(
        log_p_mature.expand_as(p_premature),
        m,
        reduction="none",
    ).mean(-1)

    kl2 = F.kl_div(
        log_p_premature,
        m,
        reduction="none",
    ).mean(-1)

    js_divs = 0.5 * (kl1 + kl2)

    # 对 batch 取平均，得到每个候选层的分数
    js_divs = js_divs.mean(-1)

    best_idx = int(js_divs.argmax().detach().cpu().item())
    best_layer = candidate_early_exit_layers[best_idx]

    js_scores = {
        candidate_early_exit_layers[i]: float(js_divs[i].detach().cpu().item())
        for i in range(len(candidate_early_exit_layers))
    }

    return best_layer, js_scores

def patch_med_opera(model, *, patch_generate: bool = True):
    """Attach Med-OPERA to one model instance without overriding model.generate.

    `model.med_opera_beam_search` is attached only to this model object.
    If `patch_generate=True`, patch HF GenerationMixin.generate globally instead of
    replacing the model's own generate(). This keeps LLaVA/LLaVA-Med multimodal
    preprocessing intact.
    """
    model.med_opera_beam_search = types.MethodType(med_opera_beam_search, model)
    model.djt_lcd_greedy_search = types.MethodType(djt_lcd_greedy_search,model)
    if patch_generate:
        install_med_opera_generation_patch()

    return model

def get_base_model(model):
    """
    兼容 LLaMA / LLaVA 一类结构：
    self.model.layers
    self.model.model.layers
    """
    base_model = getattr(model, "model", model)

    if not hasattr(base_model, "layers") and hasattr(base_model, "model"):
        base_model = base_model.model

    if not hasattr(base_model, "layers"):
        raise AttributeError("Cannot find decoder layers. Expected model.layers or model.model.layers.")

    return base_model

def hidden_to_logits(model, base_model, hidden_states, exit_layer: int):
    """
    exit_layer 是 1-based。
    exit_layer = 1 表示只经过第 1 层 transformer block。
    对应 hidden_states[1]。

    对 LLaMA 类模型：
    hidden_states[0] = embedding output
    hidden_states[1] = 第 1 层输出
    hidden_states[2] = 第 2 层输出
    ...
    """

    h = hidden_states[exit_layer]

    # 中间层 hidden state 通常还没过 final norm，所以这里补一次 norm
    if hasattr(base_model, "norm"):
        h = base_model.norm(h)
    elif hasattr(base_model, "final_layernorm"):
        h = base_model.final_layernorm(h)

    # lm_head 一般在最外层 causal LM 上
    if hasattr(model, "lm_head"):
        logits = model.lm_head(h)
    elif hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        logits = model.language_model.lm_head(h)
    else:
        raise AttributeError("Cannot find lm_head.")

    return logits[:, -1, :]

def djt_lcd_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        # djt_lcd 自定义参数
        jointed_cd_layer: Optional[int] = 8,
        boost_factor: Optional[float] = 0.5,
        relative_top: Optional[float] = 0.1,

        # 保留这些参数只是为了兼容你原来的调用接口；这里不用
        key_position: Optional[dict] = None,
        scale_factor: Optional[float] = 50.0,
        window_size: Optional[int] = 512,
        penalty_weights: Optional[float] = 1.0,

        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False  # used by synced_gpus only
    

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        base_model = get_base_model(self)
        for layer in base_model.layers:
            layer.self_attn.med_opera_boost_factor = boost_factor

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        
        # djt_lcd 必须拿 hidden_states，所以这里强制 output_hidden_states=True
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        

        if outputs.hidden_states is None:
            raise ValueError("djt_lcd greedy decode requires output_hidden_states=True.")

        # mature layer logits：最终层 logits
        final_logits = outputs.logits[:, -1, :]

        # premature layer logits：jointed_cd_layer 对应的中间层 logits
        premature_logits = hidden_to_logits(
            model=self,
            base_model=base_model,
            hidden_states=outputs.hidden_states,
            exit_layer=jointed_cd_layer,
        )

        # DoLa-style relative top filtering
        if relative_top is not None and relative_top > 0.0:
            final_logits = self.relative_top_filter(
                final_logits,
                relative_top=relative_top,
            )

            premature_logits = premature_logits.log_softmax(dim=-1)

            # final_logits 被过滤掉的位置，premature_logits 也同步压低
            mask = final_logits < -1e3
            premature_logits = premature_logits.masked_fill(mask, -1e3)

        # djt_lcd 核心：final logits 和未增强路径对应的中间层 logits 做 contrast
        next_token_logits = final_logits - premature_logits

        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished and not synced_gpus:
                break
    # print(input_ids)
    if streamer is not None:
        streamer.end()
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids