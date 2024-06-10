from transformers import GPT2Config, GPT2LMHeadModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name_or_path",
    default="openai-community/gpt2",
    help="The local or remote path to the GPT-2 model",
)
args = parser.parse_args()
gpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

# Dump the model tensor information
state = gpt.state_dict()
for k, v in state.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
