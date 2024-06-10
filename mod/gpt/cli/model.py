from transformers import GPT2Config, GPT2LMHeadModel
import torch

gpt = GPT2LMHeadModel.from_pretrained("/mnt/valerie/models/openai/gpt2")
state = gpt.state_dict()
for k, v in state.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
