import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, AutoPeftModel, AutoPeftModelForCausalLM
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from transformers import LlamaForCausalLM
from prompt.model.modeling_llama_custom import LlamaForCausalLM as CustomLlamaForCausalLM

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from contextlib import contextmanager
from prompt.utils import *
from prompt.model.model import PromptDecoder, AutoPromptDecoder, PromptConfig
from prompt.model.kv_cache import *

from pprint import pprint
from prompt.inference.dynamic_sparse_trees_3_vicuna_7b import *
 
candidate_lists = dynamic_sparse_trees_60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time

@contextmanager
def timed(wall_times, key):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def main():

    model = AutoPromptDecoder.from_pretrained(
        "hmarkc/ppd-vicuna-7b-v1.3",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.to(device)
    print(type(model))
    print(model.print_trainable_parameters())
    
    wall_times = {'init': [], 'candidates': [], 'forward_pass': [], 'evaluation': [], 'update': []}
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a big brain that is good at multi-tasking and likes prompt engineering? ASSISTANT:"
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        if not hasattr(model, "inference_buffers"):
            print("Generating buffers")
            model.generate_dynamic_buffers(candidate_lists)
        (past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        with timed(wall_times, 'init'):
            logits, prompt_logits = model.start_inference(input_ids, past_key_values, current_length_data)
        new_token = 0
        accept_lengths = []
    print(model.tokenizer.batch_decode(logits.argmax(-1)))
    
    temperature = 0.0
    posterior_threshold = 0.09
    posterior_alpha = 0.3
    sampling = 'greedy'
    
    kv_cache_lengths = []
    latencies = []
    for _ in range(512):
      with torch.inference_mode():
        with timed(wall_times, 'candidates'):
          candidates, tree_candidates_embeds = model.generate_candidates(
            logits, 
            prompt_logits, 
            temperature, 
            posterior_threshold, 
            posterior_alpha, 
            sampling)
        kv_cache_lengths.append(past_key_values[0][0].shape[2])
        with timed(wall_times, 'forward_pass'):
          logits, all_logits = model.tree_decoding(tree_candidates_embeds, past_key_values, input_ids)
        latencies.append(wall_times['forward_pass'][-1])
        with timed(wall_times, 'evaluation'):
          best_candidate, accept_length = model.evaluate_posterior(
            logits, 
            candidates, 
            temperature, 
            posterior_threshold, 
            posterior_alpha,
            sampling)
        accept_lengths.append(accept_length.cpu().item()+1)
        with timed(wall_times, 'update'):
                input_ids, logits, prompt_logits, new_token = model.update_inference_inputs(
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        logits,
                        all_logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                )
        torch.cuda.empty_cache()
        if model.tokenizer.eos_token_id in input_ids[0, :].tolist():
          break
    print(model.tokenizer.decode(input_ids[0], spaces_between_special_tokens=False,))

if __name__ == "__main__":
    main()

