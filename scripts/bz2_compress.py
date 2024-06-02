import torch
import bz2
import os
import pprint

def compress_tensor(filepath: str):
    integers = []

    # Read the file and filter out non-integer lines
    with open(filepath, 'r') as file:
        for line in file:
            try:
                number = int(line.strip())
                integers.append(number)
            except ValueError:
                # Ignore lines that cannot be converted to an integer
                continue

    # Convert list of integers to a torch tensor
    tensor = torch.tensor(integers, dtype=torch.int32)

    # Convert the tensor to bytes
    tensor_bytes = tensor.numpy().tobytes()

    # Compress the tensor bytes using bz2
    compressed_tensor = bz2.compress(tensor_bytes)

    # Save the compressed tensor to a file with .bz extension
    compressed_file_path = filepath + '.bz'
    with open(compressed_file_path, 'wb') as compressed_file:
        compressed_file.write(compressed_tensor)

# write code that gets all the files in the zipped directory
# if the file is a .txt file, compress it using the compress_tensor function
# then you need to return a dictionary where the keys are the file names and the values are file sizes of the compressed files
# the file sizes should be in bytes

def compress_files_in_directory(directory: str) -> dict:
    compressed_files = {}

    for file in os.listdir(directory):
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            compress_tensor(file_path)
            compressed_file_path = file_path + '.bz'
            compressed_file_size = os.path.getsize(compressed_file_path)
            compressed_files[file] = compressed_file_size

    return compressed_files

# write a function that deletes all files with .bz extension
def delete_compressed_files(directory: str):
    for file in os.listdir(directory):
        if file.endswith('.bz'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)

"""
{'data-models-Llama-2-70b-hf--enwik10mb_64_r16_4bit_512_16_16bit.txt': 12288,
 'data-models-Llama-2-70b-hf--enwik10mb_64_r16_4bit_512_16_8bit.txt': 971564,
 'meta-llama-Llama-2-7b-hf-enwik10mb_16_r16_512_16.txt': 1316540,
 'meta-llama-Llama-2-7b-hf-enwik10mb_1_r16_512_16.txt': 1365232,
 'meta-llama-Llama-2-7b-hf-enwik10mb_4_r16_512_16.txt': 1347663,
 'meta-llama-Llama-2-7b-hf-enwik10mb_64_r16_512_16.txt': 1285457,
 'meta-llama-Llama-2-7b-hf-enwik10mb_64_r8_512_16.txt': 1286051,
 'meta-llama-Llama-2-7b-hf-enwik16mb_64_r16_512_16.txt': 2142860,
 'meta-llama-Llama-2-7b-hf-enwik4mb_64_r16_512_16.txt': 525380,
 'meta-llama-Llama-2-7b-hf-enwik64mb_64_r16_512_16.txt': 8604095,
 'meta-llama-Llama-2-7b-hf-enwik8_64_r16_512_16.txt': 12924108,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_16_r16_512_16.txt': 1393999,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_16_r8_512_16.txt': 1366287,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_1_r16_512_16.txt': 1454927,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_1_r8_512_16.txt': 1453631,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_4_r16_512_16.txt': 1432533,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_4_r8_512_16.txt': 1418580,
 'meta-llama-Meta-Llama-3-8B-enwik10mb_64_r16_512_16.txt': 1373785,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_16_r16_512_16.txt': 1802940,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_16_r8_512_16.txt': 1892409,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_1_r16_512_16.txt': 1427681,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_1_r8_512_16.txt': 1428349,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_4_r16_512_16.txt': 1466397,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_4_r8_512_16.txt': 1468283,
 'mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_64_r8_512_16.txt': 2774420}
"""

# can you programatically figure out the base dataset it was trained on?
# you have to find the string 'enwik' in the file name and then extract the number that comes after it
# if the number is 4, it means it was trained on the enwik4mb dataset (4mb)
# if the number is 8, it means it was trained on the enwik8 dataset (100mb)
# if the number is 10, it means it was trained on the enwik10mb dataset (10mb)
# if the number is 16, it means it was trained on the enwik16mb dataset (16mb)
# if the number is 64, it means it was trained on the enwik64mb dataset (64mb)
# once you figure out the base dataset, compute the compression ratio of the compressed file
# the compression ratio is the size of the compressed file divided by the size of the base dataset
# return a dictionary where the keys are the file names and the values are the compression ratios
# if it's 4mb, then size is 4 * 10^6 bytes

def get_base_dataset(file_name: str) -> str:
    base_dataset = None
    if 'enwik4mb' in file_name:
        base_dataset = 'enwik4mb'
    elif 'enwik8' in file_name:
        base_dataset = 'enwik8'
    elif 'enwik10mb' in file_name:
        base_dataset = 'enwik10mb'
    elif 'enwik16mb' in file_name:
        base_dataset = 'enwik16mb'
    elif 'enwik64mb' in file_name:
        base_dataset = 'enwik64mb'
    return base_dataset

def get_compression_ratios(compressed_files: dict) -> dict:
    compression_ratios = {}
    for file_name, compressed_file_size in compressed_files.items():
        base_dataset = get_base_dataset(file_name)
        if base_dataset:
            if base_dataset == 'enwik4mb':
                base_dataset_size = 4 * 10**6
            elif base_dataset == 'enwik8':
                base_dataset_size = 10**8
            elif base_dataset == 'enwik10mb':
                base_dataset_size = 10 * 10**6
            elif base_dataset == 'enwik16mb':
                base_dataset_size = 16 * 10**6
            elif base_dataset == 'enwik64mb':
                base_dataset_size = 64 * 10**6
            compression_ratio = compressed_file_size / base_dataset_size
            compression_ratios[file_name] = compression_ratio
    return compression_ratios

if __name__ == "__main__":
    # Path to the file containing integers
    # filepath = 'zipped/meta-llama-Llama-2-7b-hf-enwik4mb_64_r16_512_16.txt'
    # compress_tensor(filepath)
    data = compress_files_in_directory('zipped')
    pprint.pprint(data)
    # delete_compressed_files('zipped/zipped')

    compression_ratios = get_compression_ratios(data)
    pprint.pprint(compression_ratios)

