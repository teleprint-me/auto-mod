from transformers import GPT2Config, GPT2LMHeadModel
import torch
import argparse
import texttable

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name_or_path",
    default="openai-community/gpt2",
    help="The local or remote path to the GPT-2 model",
)
parser.add_argument(
    "-t",
    "--tensors",
    action="store_true",
    help="Dump the tensor metadata to standard output",
)
parser.add_argument(
    "-n",
    "--tensor-name",
    help="Dump the tensor view to standard output by name.",
)
args = parser.parse_args()

print("Model:", args.model_name_or_path)
gpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
state = gpt.state_dict()

# Dump the model tensor information
if args.tensors:
    print("Tensor Information:")
    table = texttable.Texttable()
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            print(f"k: {k}", "|", f"v.shape: {v.shape}")

# Dump tensor information by name
if args.tensor_name:
    # e.g. positional embeddings as 'transformer.wpe.weight'
    print(state[args.tensor_name])
