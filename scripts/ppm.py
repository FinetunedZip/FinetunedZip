import pyppmd
import torch

def compress(data):
    # compressor = pyppmd.PpmdCompressor()
    return pyppmd.compress(data)

def decompress(data):
    # decompressor = pyppmd.PpmdDecompressor()
    return pyppmd.decompress(data)

def compress_file(input_file_path, compressed_file_path):
    with open(input_file_path, 'r') as file:
        original_msg = file.read()
    compressed_data = compress(original_msg)
    with open(compressed_file_path, 'wb') as f:
        f.write(compressed_data)

def compress_rank_file(input_file_path, compressed_file_path):
    with open(input_file_path, 'r') as f:
        ranks = []
        for line in f:
            ranks.append(float(line))
    ranks = torch.tensor(ranks)
    compressed_data = compress(ranks.numpy().tobytes())
    with open(compressed_file_path, 'wb') as f:
        f.write(compressed_data)

if __name__ == "__main__":
    # data = "This is a test message to be compressed."
    # compressed_data = compress(data)
    # print("Compressed data:", compressed_data)

    # decompressed_data = decompress(compressed_data)
    # print("Decompressed data:", decompressed_data)

    # input_file_path = "datasets/enwik4mb.txt"
    # compressed_file_path = "ppmd.txt"

    # compress_file(input_file_path, compressed_file_path)
    # print("File compressed successfully")

    input_file_path = "zipped/meta-llama-Llama-2-7b-hf-enwik10mb_64_r16_512_16.txt"
    compressed_file_path = "rank.ppmd"

    compress_rank_file(input_file_path, compressed_file_path)
