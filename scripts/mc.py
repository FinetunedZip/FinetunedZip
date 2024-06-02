import brotli
import torch
import bz2

def compress_ranks(filename):
    ranks = []
    with open(filename, 'r') as f:
        for line in f:
            ranks.append(float(line))
    ranks = torch.tensor(ranks)
    return brotli.compress(ranks.numpy().tobytes(), quality=11)

def save_compressed_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

def decompress_ranks(compressed_filename):
    with open(compressed_filename, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = brotli.decompress(compressed_data)
    ranks = torch.frombuffer(decompressed_data, dtype=torch.float32)
    return ranks

def verify_ranks(original_filename, decompressed_ranks):
    original_ranks = []
    with open(original_filename, 'r') as f:
        for line in f:
            original_ranks.append(float(line))
    original_ranks = torch.tensor(original_ranks)
    return torch.allclose(original_ranks, decompressed_ranks)

if __name__ == "__main__":
    input_filename = "zipped/meta-llama-Llama-2-7b-hf-enwik10mb_64_r16_512_16.txt"
    compressed_filename = "rank.br"

    # Compress the ranks and save to a file
    compressed_data = compress_ranks(input_filename)
    save_compressed_file(compressed_filename, compressed_data)

    # Decompress the ranks and verify
    decompressed_ranks = decompress_ranks(compressed_filename)
    is_verified = verify_ranks(input_filename, decompressed_ranks)

    print("Verification result:", is_verified)
