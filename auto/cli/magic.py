import struct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file-path", type=str, default="my_binary_file.bin")
parser.add_argument("-m", "--magic-value", type=str, default="AUTO")
args = parser.parse_args()

# Define your MAGIC constant
MAGIC = int.from_bytes(args.magic_value.encode(), byteorder="little", signed=False)


def write_magic(filename):
    # Open the output file in binary mode
    with open(filename, "wb") as f:
        # Write the MAGIC value to the file using struct.pack()
        f.write(struct.pack("<I", MAGIC))


# Call write_magic function and specify a filename
write_magic(args.file_path)
