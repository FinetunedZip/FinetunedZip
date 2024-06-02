# write a script that gets all the files in the zipped directory with the extension .gpz
# print out their "compression ratios" which is their file size in bytes over 10^7. 

import os

def get_files(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".gpz"):
            files.append("zipped/" + file)
    return files

def get_file_size(file):
    return os.path.getsize(file)

def get_compression_ratio(file, size=64 * 10 ** 6):
    return get_file_size(file) / size

def main():
    directory = "zipped"
    files = get_files(directory)
    for file in files:
        # if "enwik4mb" in file:
        #     print("file_size: ", get_file_size(file))
        #     # print(f"{file}: {get_compression_ratio(file, size=4*10**6)}")
        # if "enwik16mb" in file:
        #     print(f"{file}: {get_compression_ratio(file, size=16*10**6)}")
        # if "enwik64mb" in file:
        #     print(f"{file}: {get_compression_ratio(file, size=64*10**6)}")
        # else: 
        print(f"{file}: {get_compression_ratio(file)}")
        # print(f"file_size of {file}: ", get_file_size(file))

if __name__ == "__main__":
    main()

    # python3 -m scripts.filesize