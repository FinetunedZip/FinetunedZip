import brotli
import bz2

def compress_with_brotli_file(filename):
    text = ""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return brotli.compress(text.encode('utf-8'), quality=11)

# make a function that compresses text with bz2
def compress_with_bz2_file(filename):
    text = ""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return bz2.compress(text.encode('utf-8'))

def save_compressed_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

# write a script that takes in a file of "numbers"
# each number is on a new line. convert the file to a torch array
# convert the array to bytes and compress it with brotli
        
def compress_ranks(filename):
    import torch
    ranks = []
    with open(filename, 'r') as f:
        for line in f:
            ranks.append(float(line))
    ranks = torch.tensor(ranks)
    return bz2.compress(ranks.numpy().tobytes())


if __name__ == "__main__":
    # data = compress_with_brotli_file("zipped/meta-llama-Llama-2-7b-hf-enwik8_64_r16_512_16.txt")
    # save_compressed_file("rank.br", data)
    # data = compress_with_bz2_file("zipped/meta-llama-Llama-2-7b-hf-enwik8_64_r16_512_16.txt")
    # save_compressed_file("rank.bz2", data)

    # # print the size of the compressed files
    # import os
    # print("Size of compressed files")
    # print("rank.br", os.path.getsize("rank.br"))
    # print("rank.bz2", os.path.getsize("rank.bz2"))

    data = compress_ranks("zipped/meta-llama-Llama-2-7b-hf-enwik4mb_64_r16_512_16.txt")
    save_compressed_file("rank.br", data)