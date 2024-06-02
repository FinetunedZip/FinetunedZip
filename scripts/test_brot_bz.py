import brotli
import bz2
import os
import pyppmd

# write a script that tests out brotli compression on 4 different datasets
datasets = ["datasets/enwik4mb.txt", "datasets/enwik16mb.txt", "datasets/enwik64mb.txt", "datasets/enwik8.txt"]

# store the compression ratios which is the size of the compressed file divided by the size of the original file
def test_brot_bz(datasets):
    for dataset in datasets:
        # read the data
        with open(dataset, 'r') as file:
            data = file.read()
        
        # compress the data using brotli and bz2
        compressed_brotli = brotli.compress(data.encode())
        compressed_bz2 = bz2.compress(data.encode())
        compressed_ppmd = pyppmd.compress(data.encode())
        
        # write the compressed data to a new file
        with open(f"compressed_brotli_{os.path.basename(dataset)}", 'wb') as file:
            file.write(compressed_brotli)
        
        with open(f"compressed_bz2_{os.path.basename(dataset)}", 'wb') as file:
            file.write(compressed_bz2)

        with open(f"compressed_ppmd_{os.path.basename(dataset)}", 'wb') as file:
            file.write(compressed_ppmd)
        
        # calculate the compression ratio
        compression_ratio_brotli = len(compressed_brotli) / len(data)
        compression_ratio_bz2 = len(compressed_bz2) / len(data)
        compression_ratio_ppmd = len(compressed_ppmd) / len(data)
        
        print(f"Dataset: {os.path.basename(dataset)}")
        print(f"Compression ratio (brotli): {compression_ratio_brotli}")
        print(f"Compression ratio (bz2): {compression_ratio_bz2}")
        print(f"Compression ratio (ppmd): {compression_ratio_ppmd}")
        print()

# can you make a function that executes every variation of double compression possible?
# this means that you compress the data using brotli and then compress the compressed data using bz2
# only do this for the enwik8 dataset
        
def double_compression(dataset):
    with open(dataset, 'r') as file:
        data = file.read()

    compressed_data = {
        "brotli": brotli.compress(data.encode()),
        "bz2": bz2.compress(data.encode()),
        "ppmd": pyppmd.compress(data.encode())
    }

    double_compression_ratios = {}

    for method, compressed in compressed_data.items():
        double_compressed = {
            "brotli": brotli.compress(compressed),
            "bz2": bz2.compress(compressed),
            "ppmd": pyppmd.compress(compressed)
        }
        double_compression_ratios[method] = {
            k: len(v) / len(data) for k, v in double_compressed.items()
        }

    print(f"Double Compression Ratios for {os.path.basename(dataset)}:")
    for method, ratios in double_compression_ratios.items():
        for second_method, ratio in ratios.items():
            print(f"First compression with {method}, second compression with {second_method}: {ratio}")
        print()

if __name__ == "__main__":
    # test_brot_bz(datasets)
    double_compression("datasets/enwik8.txt")

"""

Dataset: enwik4mb.txt
Compression ratio (brotli): 0.2796320227468801
Compression ratio (bz2): 0.29154902511137615
Compression ratio (ppmd): 0.2508653103103428

Dataset: enwik16mb.txt
Compression ratio (brotli): 0.27430132713558386
Compression ratio (bz2): 0.293984198166162
Compression ratio (ppmd): 0.25080435217942726

Dataset: enwik64mb.txt
Compression ratio (brotli): 0.27180588429172897
Compression ratio (bz2): 0.29153793118043253
Compression ratio (ppmd): 0.2495868193472855

Dataset: enwik8.txt
Compression ratio (brotli): 0.271477209935268
Compression ratio (bz2): 0.2911887627202037
Compression ratio (ppmd): 0.24925631763125977

-------

Double Compression Ratios for enwik8.txt:
First compression with brotli, second compression with brotli: 0.27147805312393775
First compression with brotli, second compression with bz2: 0.27267429693523404
First compression with brotli, second compression with ppmd: 0.2779643221176659

First compression with bz2, second compression with brotli: 0.2911896862125563
First compression with bz2, second compression with bz2: 0.29256348146659256
First compression with bz2, second compression with ppmd: 0.2965947564586044

First compression with ppmd, second compression with brotli: 0.24925709055420703
First compression with ppmd, second compression with bz2: 0.250366335363116
First compression with ppmd, second compression with ppmd: 0.25526592403962217

"""