from dataclasses import dataclass, field
import warnings
import math
import pathlib
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer, BitsAndBytesConfig
#from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.tokenization_utils_base import BatchEncoding

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from prompt.utils import *
from prompt.model.model import PromptDecoder, PromptConfig, AutoPromptDecoder
from prompt.model.modeling_llama_custom import LlamaForCausalLM as CustomLlamaForCausalLM
from prompt.model.kv_cache import initialize_past_key_values
from torch.utils.data import Dataset
import sys

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

sys.stderr = open('error_log.txt', 'w')
sys.stdout = open('output.txt', 'w')

class JSONDataset(Dataset):
    def __init__(self,tensor_data):
        self.data = tensor_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def JsonDataset(dataset, tokenizer):
    tokenized_data = []
    for item in dataset:
        input_text = item['instruction']
        output_text = item['output']
        
        input_encoding = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        output_encoding = tokenizer.encode_plus(
            output_text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        tokenized_data.append({
            'input_ids': torch.tensor(input_encoding.input_ids),
            'attention_mask': torch.tensor(input_encoding.attention_mask),
            'labels': torch.tensor(output_encoding.input_ids),
            'output_attention_mask': torch.tensor(output_encoding.attention_mask)
        })
        #print(input_encoding.input_ids)
    # inputs_tensor = torch.stack([item['input_ids'] for item in tokenized_data])
    # input_masks_tensor = torch.stack([item['attention_mask'] for item in tokenized_data])
    # outputs_tensor = torch.stack([item['labels'] for item in tokenized_data])
    # output_masks_tensor = torch.stack([item['output_attention_mask'] for item in tokenized_data])
    
    # tensor_data = {
    #     'input_id': inputs_tensor,
    #     'attention_mask': input_masks_tensor,
    #     'labels': outputs_tensor,
    #     'output_masks': output_masks_tensor
    # }
    #print(tokenized_data)
    return tokenized_data

class ParamEfficientFineTuner(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        num_special_tokens = self.model.active_peft_config.num_special_tokens
        if torch.any(inputs["input_ids"][:, -1] == self.tokenizer.eos_token_id):
            warnings.warn("Input ends with EOS token.")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits

        # Calculate loss on the prompt tokens
        prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
        prompt_labels = labels[..., -num_special_tokens:].contiguous()
        prompt_labels = prompt_labels.to(logits.device)
        loss = 0
        loss_fn = CrossEntropyLoss()
        decay_coefficient = 0.8
        for i in range(num_special_tokens):
            loss += loss_fn(prompt_logits[:, i, :], prompt_labels[:, i]) * (decay_coefficient ** i)
        if num_special_tokens > 0:
            loss /= num_special_tokens
        return (loss, outputs) if return_outputs else loss


class DistillationTrainer(Trainer):
    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]

        # test time training parameters
        self.mode = args.mode
        self.eval_interval = args.ttt_eval_interval
        self.ttt_update_interval = args.ttt_update_interval
        self.buffer = []
        self.training_step_1 = True
        self.accept_length_list = []
        self.training_step_cnt = 0
        self.ttt_input_ids = torch.tensor([])
        self.ttt_logits = torch.tensor([])
        self.ttt_prompt_logits = torch.tensor([])
        self.eval_path = args.accuracy_path
        self.new_token = 0
        print("mode:", self.mode)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        #self.training_step_cnt += 1
        if self.mode == "offline":
            print("training_step: offline")
            return self.ppd_compute_loss(model, inputs, return_outputs)
        elif self.mode == "online":
            return self.ttt_compute_loss(model, inputs, return_outputs)
        else:
            raise ValueError()

    def ppd_compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        num_special_tokens = self.model.active_peft_config.num_special_tokens
        if torch.any(inputs["input_ids"][:, -1] == self.tokenizer.eos_token_id):
            warnings.warn("Input ends with EOS token.")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits

        # Calculate loss on the prompt tokens
        prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
        prompt_labels = labels.contiguous()
        prompt_labels = prompt_labels.to(logits.device)
        loss = 0
        decay_coefficient = 0.8
        for i in range(num_special_tokens):
            loss_i = F.kl_div(
                F.log_softmax(prompt_logits[:, i, :], dim=-1),
                F.softmax(prompt_labels[:, i, :], dim=-1),
                reduction='sum'
            ) / prompt_logits.shape[0]
            loss += loss_i * (decay_coefficient ** i)
        if num_special_tokens > 0:
            loss /= num_special_tokens
        return (loss, outputs) if return_outputs else loss
    
    def ttt_compute_loss(self, model, inputs, return_outputs=False):
        max_new_tokens = 128
        batch_size = inputs["input_ids"].shape[0]
        assert(
            batch_size == 1
        )
        num_special_tokens = self.model.active_peft_config.num_special_tokens

        #remove masking
        input_ids = inputs["input_ids"]
        input_ids = torch.squeeze(input_ids, dim=1)
        print("input_ids_shape_train:", input_ids.shape)
        #[inputs["attention_mask"]].unsqueeze(0)
        
        with torch.inference_mode():
            if not hasattr(self, "inference_buffers"):
                print('Generate buffers')
                model.generate_dynamic_buffers(get_dynamic_sparse_tree(model.active_peft_config.base_model_name_or_path))
            # Initialize the past key and value states
            if hasattr(self, "past_key_values"):
                past_key_values = model.past_key_values
                past_key_values_data = model.past_key_values_data
                current_length_data = model.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                print('Initialize past key values')
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(model.base_model)
                model.past_key_values = past_key_values
                print("past_key_values:", len(past_key_values))
                model.past_key_values_data = past_key_values_data
                model.current_length_data = current_length_data
            self.ttt_logits, self.ttt_prompt_logits = model.start_inference(input_ids, past_key_values, current_length_data)
            new_token = 0
            accept_lengths = []
            self.ttt_input_ids = input_ids
        
        #use ppd to generate tokens
        # if self.training_step_1:
        #     outputs = model(
        #     input_ids=input_ids,
        #     )
        #     self.ttt_logits = outputs.logits[:, -num_special_tokens-1:-num_special_tokens, :]
        #     self.ttt_prompt_logits = outputs.logits[:, -num_special_tokens:, :]
        #     self.training_step_1 = False
        #     self.ttt_input_ids = input_ids
        
        temperature = 0.0
        posterior_threshold = 0.09
        posterior_alpha = 0.3
        sampling = 'greedy'

        with torch.inference_mode():
            for _ in range(64):
                self.training_step_cnt += 1
                self.model.eval()
                candidates, tree_candidates_embeds = model.generate_candidates(
                    self.ttt_logits, 
                    self.ttt_prompt_logits, 
                    temperature, 
                    posterior_threshold, 
                    posterior_alpha, 
                    sampling)
                print("ttt_input_ids:", self.ttt_input_ids.shape)
                print("past_key_values_1:", len(past_key_values))
                logits, all_logits = model.tree_decoding(tree_candidates_embeds, past_key_values, self.ttt_input_ids)
                best_candidate, accept_length = model.evaluate_posterior(
                    logits, 
                    candidates, 
                    temperature, 
                    posterior_threshold, 
                    posterior_alpha,
                    sampling)
                #get first rejected logits from best candidates
                if accept_length != logits.size(-1):
                    test_logits = logits[None, best_candidate, accept_length+1 : accept_length + 2]
                    #get first regjected prompt logits
                    candidate_index = model.inference_buffers['retrieve_indices'][best_candidate, accept_length : accept_length + 1]
                    prompt_token_indices = model.inference_buffers['special_token_indices'][candidate_index.cpu().item()]
                    test_prompt_logits = all_logits[:, prompt_token_indices][:,0,:]
                    #update buffer with rejected candidates and logits
                    self.buffer.append([test_logits, test_prompt_logits])
                self.ttt_input_ids, self.ttt_logits, self.ttt_prompt_logits, self.new_token = model.update_inference_inputs(
                        self.ttt_input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        logits,
                        all_logits,
                        self.new_token,
                        past_key_values_data,
                        current_length_data,
                )
                #inputs["input_ids"] = input_ids
                #yield input_ids, logits, prompt_logits, new_token
                #self.accept_length_list.append(accept_length)
                #cumulative
                self.accept_length_list.append(accept_length/num_special_tokens)
                log_path = os.path.join(self.eval_path)
                os.makedirs(self.eval_path, exist_ok=True)
                if self.training_step_cnt%self.eval_interval==0:
                    #avg_accept_length = sum(self.accept_length_list)*1.0/self.eval_interval
                    #acceptance_rate = avg_accept_length/num_special_tokens
                    #self.accept_length_list = []
                    avg_accept_length = sum(self.accept_length_list)*1.0/len(self.accept_length_list)
                    with open("log/ttt/accuracy/accuracy.txt", 'a') as log_file:
                        log_file.write(f"acceptance_rate: {avg_accept_length}\n")

        #update model
        if len(self.buffer) >= self.ttt_update_interval:
            self.model.train()

            #compute loss
            loss = 0
            for logit, prompt_logit in self.buffer:
                loss_i = F.kl_div(
                    F.log_softmax(prompt_logit),
                    F.softmax(logit),
                    reduction='sum'
                ) / self.ttt_prompt_logits.shape[0]
                loss += loss_i
            #loss.backward()
            self.buffer = []
            #return loss.detach()
            loss.requires_grad = True
            return (loss, outputs) if return_outputs else loss
        else:
            self.model.eval()
            return torch.tensor(-1).cuda()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="lmsys/vicuna-7b-v1.3")
    num_special_tokens: int = field(default=1)
    virtual_tokens_per_special_token: int = field(default=1)
    use_custom_lm_head: bool = field(default=False)
    use_prefix_tuning: bool = field(default=False)
    prefix_virtual_tokens: int = field(default=10)
    vt_attention_type: str = field(default="decoder")
    aggregation_type: str = field(default="mean")
    num_exits: int = field(default=1)
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the saved dataset."},
    )
    size: Optional[int] = field(
        default=None, metadata={"help": "Number of examples to use."}
    )
    use_chunked: bool = field(
        default=False, metadata={"help": "Whether to use chunked dataset."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default="adamw_torch")
    trainer_type: str = field(default="param_efficient_fine_tuner", metadata={"help": "Trainer type: param_efficient_fine_tuner, distillation_trainer"})
    stage1_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the stage 1 model."},
    )
    lm_head_lr_multiplier: float = field(default=0.1)
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mode: str = field(
        default = "offline",
        metadata={"help": "offline training mode or online training mode"},
    )
    online_training_model_path: str = field(
        default="hmarkc/ppd-vicuna-7b-v1.3",
    )
    ttt_update_interval: int = field(
        default = 8,
        metadata={
            "help": "maximum buffer size before update the model for test time training"
        },
    )
    ttt_eval_interval: int = field(
        default = 10,
        metadata={
            "help":"Interval at which acceptance length is ploted"
        },
    )
    accuracy_path: str = field(
        default = "./log/ttt",
        metadata = {
            "help":"path to log evaluation results"
        },
    )



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("load_in_4_bits", model_args.load_in_4bit)

    # Create model
    peft_config = PromptConfig(
        tokenizer_name_or_path=model_args.model_name_or_path,
        base_model_name_or_path=model_args.model_name_or_path,
        num_special_tokens=model_args.num_special_tokens,
        virtual_tokens_per_special_token=model_args.virtual_tokens_per_special_token,
        use_prefix_tuning=model_args.use_prefix_tuning,
        prefix_virtual_tokens=model_args.prefix_virtual_tokens,
        vt_attention_type=VTAttentionType.from_str(model_args.vt_attention_type),
        aggregation_type=AggregationType.from_str(model_args.aggregation_type),
        use_custom_lm_head=model_args.use_custom_lm_head,
        num_exits=model_args.num_exits,
    )
    if training_args.stage1_model_path:
        model = AutoPromptDecoder.from_pretrained(
            training_args.stage1_model_path,
            low_cpu_mem_usage=True,
            cache_dir=training_args.cache_dir,
            quantization_config=quantization_config if model_args.load_in_4bit else None,
            new_config=peft_config,
        )
    elif training_args.mode == "online":
        model = AutoPromptDecoder.from_pretrained(
            training_args.online_training_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    else:
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.use_cache = False

        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
        
        if config.model_type == "llama":
            base_model = CustomLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config if model_args.load_in_4bit else None,
                # load_in_4bit=model_args.load_in_4bit,
                # load_in_8bit=model_args.load_in_8bit,
            )
        else:
            raise ValueError("Only support llama for now")

        for param in base_model.base_model.parameters():
            param.requires_grad = False
        model = PromptDecoder(base_model, peft_config)
    print(model.print_trainable_parameters(), model)

    # Output dir
    training_args.output_dir = f"{training_args.output_dir}/prompt_{model_args.model_name_or_path.split('/')[-1]}_{model_args.num_special_tokens}_{model_args.virtual_tokens_per_special_token}_cl{training_args.model_max_length}_{model_args.vt_attention_type.upper()}_{model_args.aggregation_type}{'_custom_lm_head' if model_args.use_custom_lm_head else ''}{'_prefix' + str(model_args.prefix_virtual_tokens) if model_args.use_prefix_tuning else ''}_exits{model_args.num_exits}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        truncation=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    
    # Load data
    # if data_args.use_chunked:
    #     data = ChunkDataset(data_args.dataset_path)
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    if training_args.mode == "online":
        train_data = json.load(open(data_args.dataset_path, "r"))
        #train_data = train_data[:10]
        data = JsonDataset(train_data, tokenizer)
        data = JSONDataset(data)
        #data.set_size(data_args.size)
        #print("dataset:"data.shape())
        #data = DataLoader(data, batch_size=1, shuffle=True)
        # inputs_id = [item['instruction'] for item in train_data]
        # labels = [item['output'] for item in train_data]
        # inputs_tensor = torch.tensor(inputs_id)
        # outputs_tensor = torch.tensor(labels)
        # data = {
        #     'inputs_id': inputs_tensor,
        #     'labels': outputs_tensor
        # }
    else:
        data = torch.load(data_args.dataset_path)
        data.set_size(data_args.size)

    # Set up optimizer 
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (p.requires_grad and "lm_head" in n)
            ],
            "lr": training_args.learning_rate * training_args.lm_head_lr_multiplier,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (p.requires_grad and "prompt_encoder" in n)
            ],
            "lr": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
        },
    ]
    optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optim_cls(optimizer_grouped_parameters, **optim_kwargs)
    
    # Start trainner
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    if training_args.trainer_type == "distillation_trainer":
        trainer = DistillationTrainer(
            model=model, tokenizer=tokenizer, args=training_args, train_dataset=data, eval_dataset=None, optimizers=(optimizer, None)
        )
    elif training_args.trainer_type == "param_efficient_fine_tuner":
        trainer = ParamEfficientFineTuner(
            model=model, tokenizer=tokenizer, args=training_args, train_dataset=data, eval_dataset=None, optimizers=(optimizer, None)
        )
    else: 
        raise ValueError(f"Trainer type {training_args.trainer_type} not supported.")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()