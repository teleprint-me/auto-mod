from transformers import GPT2Config, GPT2LMHeadModel
import torch
import argparse
import texttable


def print_table(table_data: list[list[str]], width: int = 240) -> None:
    """Prints a formatted table of the CSV contents to the console.

    Args:
        table_data: The table to print.
        width (optional): The width of the table. Defaults to 240.
    """
    tt = texttable.Texttable(width)
    tt.set_deco(texttable.Texttable.HEADER)
    tt.set_cols_dtype(["t"] * len(table_data[0]))
    tt.set_cols_align(["l"] * len(table_data[0]))
    tt.add_rows(table_data)
    print(tt.draw())


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path",
        default="openai-community/gpt2",
        help="The local or remote path to the GPT-2 model",
    )

    parser.add_argument(
        "-s",
        "--shape",
        action="store_true",
        help="Dump the tensor metadata to standard output",
    )
    parser.add_argument(
        "-v",
        "--view",
        help="Dump the tensor view to standard output by name.",
    )

    parser.add_argument(
        "-a", "--architecture", action="store_true", help="View the architecture"
    )
    parser.add_argument(
        "-m", "--model", action="store_true", help="View the model structure"
    )
    parser.add_argument("-l", "--layers", help="List the block ids")
    parser.add_argument("-b", "--blocks", help="List the block ids")
    parser.add_argument("-i", "--block-id", help="View a block using a block id")

    return parser.parse_args()


args = get_arguments()

print("Model:", args.model_name_or_path)
# Get the model
gpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
# Get the models state
state = gpt.state_dict()

# Dump the model tensor shapes
if args.shape:
    print("Tensor Information:")
    table = texttable.Texttable()
    model = [[k, v.shape] for k, v in state.items() if isinstance(v, torch.Tensor)]
    model.insert(0, ["Tensor Name", "Tensor Shape"])
    print_table(model)

# Dump tensor information by name
if args.view:
    # e.g. positional embeddings as 'transformer.wpe.weight'
    tensor = state[args.tensor_name]
    n, m = args.view.split(",")
    # get the final view
    print(tensor.view(int(n), int(m)))

# Dump the model architecture
if args.architecture:
    print(gpt)

if args.model:
    print(gpt.transformer)
