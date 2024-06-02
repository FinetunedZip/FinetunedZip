import io
import json
import random
import torch
import zlib
import array
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM
import numpy as np
import matplotlib.pyplot as plt
from pyfastpfor import getCodec
import bz2
import zlib
import brotli

def tensor_to_numpy(tensor):
    # Ensure the tensor is on CPU and convert it to NumPy
    conv = np.array(tensor.cpu().numpy(), dtype=np.uint32, order='C')
    # ensure that eaach value is uint32
    for i in range(len(conv)):
        conv[i] = np.uint32(conv[i])
    return conv

def numpy_to_tensor(array: np.ndarray, device):
    # Convert the NumPy array back to a PyTorch tensor
    # convert the type of the np array to float32
    array = array.astype(np.float32)
    return torch.tensor(array, device=device)

class MLMZip():
    def __init__(self, model_name, tokenizer_name, model, tokenizer, finetuned, context_size, batch_size):
        self.CONTEXT_SIZE = context_size
        self.BATCH_SIZE = batch_size
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.finetuned = finetuned

        self.device = torch.cuda.current_device()
        self.model.to(self.device)

        # self.frequency_mapping = json.load(open("datasets/data_100MB_mapping_key.json", "r"))
        # self.reverse_frequency_mapping = json.load(open("datasets/reverse_mapping.json", "r"))
        self.frequency_mapping = json.load(open("datasets/filled_map.json", "r"))
        self.reverse_frequency_mapping = json.load(open("datasets/reverse_filled_mapping.json", "r"))

        self.mapped_mask = []

        self.tokens = []
        self.ranks = []

    def text_to_tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)["input_ids"].squeeze().to(self.device)
        print("DONE TOKENIZING")
        return tokens

    def tokens_to_text(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def create_deterministic_masks(self, token_length):
        random.seed(0)  # Set a fixed seed for the random number generator
        mask = [False] * token_length
        num_to_mask = int(token_length * 0)  # Calculate the number of tokens to mask based on the desired masking percentage
        indices_to_mask = random.sample(range(token_length), num_to_mask)  # Randomly select indices to mask
        for i in indices_to_mask:
            mask[i] = True
        return torch.tensor(mask, dtype=torch.bool)

    def mask_tokens(self, tokens):
        token_length = tokens.size(0)
        # print(token_length)
        masks = self.create_deterministic_masks(token_length)
        masked_tokens = tokens.clone()
        masked_tokens[masks] = self.tokenizer.mask_token_id
        return masked_tokens, masks

    def get_predictions(self, masked_tokens):
        masked_tokens = masked_tokens.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(masked_tokens) # auto does batching
        logits = outputs.logits
        return logits

    def replace_masked_tokens(self, logits, masked_tokens, masks, original_tokens):
        # Get the model's predictions for each token
        predictions = torch.argsort(logits, dim=-1, descending=True).squeeze()

        ranks = torch.full_like(masked_tokens, -1)

        # print(logits.shape)
        for i in range(logits.shape[1]):  # iterate over sequence length
            if masks[i]:
                ranks[i] = (predictions[i] == original_tokens[i]).nonzero(as_tuple=True)[0]

        # figure out how many ranks are between 0 and 15
        self.ranks.extend(ranks.cpu().numpy())
        # print(self.ranks)
        # print("RANKS BETWEEN 0-15:", len([r for r in self.ranks if r < 16 and r >= 0]) / len([r for r in self.ranks if r >= 0]) * 100, "%")
        # print("NUM RANKS:", len([r for r in self.ranks if r >= 0]))
        # Replace the masked tokens with the ranks of the correct tokens
        masked_tokens[masks] = ranks[masks].to(masked_tokens.device)
        # print(masked_tokens)
        return masked_tokens

    def encode_and_zip(self, text):
        tokens = self.text_to_tokens(text)
        print(len(tokens))
        self.tokens = tokens
        cnt = 0
        token_chunks = tokens.split(self.CONTEXT_SIZE)
        decoded_tokens_list = []
        total_chunks = len(token_chunks)
        map_flag = False
        map_matrix = []

        with tqdm(total=total_chunks) as pbar:
            for i, token_chunk in enumerate(token_chunks):
                original_tokens = token_chunk.clone()
                masked_tokens, masks = self.mask_tokens(token_chunk)
                logits = self.get_predictions(masked_tokens)
                decoded_tokens = self.replace_masked_tokens(logits, masked_tokens, masks, original_tokens)
                for i in range(len(decoded_tokens)):
                    if masks[i] == False and str(decoded_tokens[i].item()) in self.frequency_mapping and decoded_tokens[i].item() > self.frequency_mapping[str(decoded_tokens[i].item())]:
                        decoded_tokens[i] = torch.tensor(self.frequency_mapping[str(decoded_tokens[i].item())], device=self.device)
                        cnt += 1
                        map_matrix.append(1)
                    else:
                        map_matrix.append(0)

                # print("MAP MATRIX SIZE", len(map_matrix))
                # print("TOKEN SIZE", len(decoded_tokens))

                decoded_tokens_list.append(decoded_tokens)
                self.mapped_mask.extend(map_matrix)
                map_matrix = []
                pbar.update(1)
                pbar.set_description(f"Encoding Progress: {((i+1)/total_chunks)*100:.2f}%")

        # Concatenate all decoded tokens into a single tensor
        all_decoded_tokens = torch.cat(decoded_tokens_list)
        map_mask = torch.tensor(self.mapped_mask, dtype=torch.float32, device=self.device)

        # count how many 0's are in map_mask and print it out
        # print(len([m for m in map_mask if m == 0]), "0's in map_mask")

        # print(all_decoded_tokens.size())
        # print("REPLACED", cnt, "TOKENS")

        # zip_package = torch.stack([all_decoded_tokens, map_mask])
        # concatenate all_decoded_tokens and map_mask into a single tensor with shape (1, num_tokens * 2)
        zip_package = torch.cat([all_decoded_tokens, map_mask])
        print(zip_package.size())
        # exit(1)

        # buffer = io.BytesIO()
        # torch.save(zip_package, buffer)
        # buffer.seek(0)
        # tensor_bytes = buffer.read()
        # zipped_data = zlib.compress(tensor_bytes, level=9)

        # tensor_np = tensor_to_numpy(zip_package)
        # print(tensor_np[:5])

        # Compress data using the new codec
        codec = getCodec('optpfor')
        inpComp = np.zeros(tensor_np.size, dtype=np.uint32, order='C')
        compSize = codec.encodeArray(tensor_np, tensor_np.size, inpComp, len(inpComp))
        # print(compSize)
        # print(inpComp[:compSize])
        print("PRE: ", inpComp.size)
        print(compSize)

        compressed_array = inpComp[:compSize]
        # Convert compressed data back to bytes for storage
        compressed_bytes = compressed_array.tobytes()
        further_compressed = self.compress_with_brotli(compressed_bytes)

        return further_compressed


        # return zipped_data

    def unzip_and_decode(self, zipped_data):
        # Decompress the data to get back the byte representation of the tensor
        zipped_data = self.decompress_brotli(zipped_data)

        codec = getCodec('optpfor')
        # print(len(zipped_data) //  4)
        compSize = len(zipped_data) // 4  # Assuming uint32 for each element
        # we converted inpComp into bytes, so we need to convert it back to uint32
        inpComp = np.frombuffer(zipped_data, dtype=np.uint32)
        # print(inpComp)
        print(inpComp.size)
        print(compSize)
        print(compSize * 4)

        # Decompress data
        inpCompDecomp = np.zeros(len(inpComp) * 6, dtype=np.uint32)  # Update arrSize accordingly
        size = codec.decodeArray(inpComp, compSize, inpCompDecomp, len(inpComp) * 6)
        print("DECODED: ", size)
        # Convert numpy array back to torch.Tensor
        loaded_tensor = numpy_to_tensor(inpCompDecomp, self.device)
        loaded_tensor = loaded_tensor[:size]

        # Use an in-memory bytes buffer to reconstruct the tensor
        # buffer = io.BytesIO(tensor_bytes)
        # buffer.seek(0)  # Move to the start of the buffer
        # loaded_tensor = torch.load(buffer)
        print("LOADED TENSOR")
        print(loaded_tensor.size())
        # exit(1)

        # print(loaded_tensor[0].size())
        # print(loaded_tensor[1].size())
        # exit(1)
        # tensor = loaded_tensor[0]
        # map_mask = loaded_tensor[1]

        # convert all the values in loaded_tensor to int
        loaded_tensor = loaded_tensor.to(torch.int32)

        # split the loaded_tensor into 2 tensors of equal length (split in half)
        tensor, map_mask = torch.chunk(loaded_tensor, 2)

        # Split the tensor into chunks of size self.CONTEXT_SIZE
        tensor_chunks = tensor.split(self.CONTEXT_SIZE)
        map_mask = map_mask.split(self.CONTEXT_SIZE)
        cnt = 0

        # all_tokens = torch.tensor()

        decoded_text_chunks = []
        # print(tensor_chunks[0][:50])
        for j in range(len(tensor_chunks)):

            # Generate mask for the tensor chunk
            mask = self.create_deterministic_masks(tensor_chunks[j].size(0))

            for i in range(len(tensor_chunks[j])):
                if map_mask[j][i] == 1:
                    tensor_chunks[j][i] = torch.tensor(self.reverse_frequency_mapping[str(tensor_chunks[j][i].item())], device=self.device)
                    cnt += 1

            masked_tokens = tensor_chunks[j].clone()
            masked_tokens[mask] = self.tokenizer.mask_token_id

            masked_tokens = masked_tokens.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(masked_tokens)
            logits = outputs.logits
            predictions = torch.argsort(logits, dim=-1, descending=True).squeeze()

            ranks = torch.zeros_like(tensor_chunks[j])
            for i in range(logits.shape[1]):
                if mask[i]:
                    ranks[i] = predictions[i][tensor_chunks[j][i]]

            tensor_chunks[j][mask] = ranks[mask].to(tensor_chunks[j].device)

            # all_tokens.append(tensor_chunks[j])
            decoded_text_chunk = self.tokenizer.decode(tensor_chunks[j], skip_special_tokens=True)
            decoded_text_chunks.append(decoded_text_chunk)

        print("REPLACED", cnt, "TOKENS")
        # Concatenate the decoded text chunks
        decoded_text = ' '.join(decoded_text_chunks)
        decoded_text = decoded_text.replace(' ##', '')

        return decoded_text

    def zip_file(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # print(text)
        zipped_data = self.encode_and_zip(text)
        with open(output_file, 'wb') as f:
            f.write(zipped_data)

    def unzip_file(self, input_file, output_file):
        with open(input_file, 'rb') as f:
            zipped_data = f.read()
        decoded_text = self.unzip_and_decode(zipped_data)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_text)

    def compress_with_brotli(self, byte_data):
        return brotli.compress(byte_data, quality=11)

    def decompress_brotli(self, compressed_data):
        return brotli.decompress(compressed_data)

    def compress_with_bz2(self, byte_data):
        compressor = bz2.BZ2Compressor()
        compressed_data = compressor.compress(byte_data)
        compressed_data += compressor.flush()
        return compressed_data

    def plot_rank_distribution(self, plot_type='histogram'):
        if plot_type == 'histogram':
            plt.figure(figsize=(10, 6))
            plt.hist(self.ranks, bins=100, log=True)
            plt.title('Distribution of Token Ranks')
            plt.xlabel('Rank')
            plt.ylabel('Frequency (log scale)')
        elif plot_type == 'scatter':
            plt.figure(figsize=(10, 6))
            x = list(range(len(self.ranks)))
            y = self.ranks
            plt.scatter(x, y, alpha=0.5)
            plt.title('Scatter Plot of Token Ranks')
            plt.xlabel('Sequence Position')
            plt.ylabel('Rank')
            plt.yscale('log')
        else:
            print("Invalid plot type specified. Please choose 'histogram' or 'scatter'.")

        plt.show()

if __name__ == "__main__":
    torch.cuda.set_device(0)
    model_name = "google-bert/bert-base-uncased"
    tokenizer_name = "google-bert/bert-base-uncased"
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    mlmzip = MLMZip(model_name, tokenizer_name, model, tokenizer, False, 512, 16)
    # mlmzip.zip_file("datasets/data.txt", "text.gpz")
    # mlmzip.unzip_file("text.gpz", "datasets/data_decoded.txt")

    # with open("datasets/data.txt", 'r', encoding='utf-8') as f:
    #     text = f.read()
    # with open("datasets/data_decoded.txt", 'r', encoding='utf-8') as f:
    #     text2 = f.read()

    mlmzip.zip_file("datasets/1mb.txt", "1mb_text.gpz")
    mlmzip.unzip_file("1mb_text.gpz", "datasets/1mb_decoded.txt")

    with open("datasets/1mb.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    with open("datasets/1mb_decoded.txt", 'r', encoding='utf-8') as f:
        text2 = f.read()

    # mlmzip.zip_file("datasets/10mb.txt", "10mb_text.gpz")
    # mlmzip.unzip_file("10mb_text.gpz", "datasets/10mb_decoded.txt")

    # with open("datasets/10mb.txt", 'r', encoding='utf-8') as f:
    #     text = f.read()
    # with open("datasets/10mb_decoded.txt", 'r', encoding='utf-8') as f:
    #     text2 = f.read()

    # print("TEXT 2")
    # print(text2[:100])

    # print(mlmzip.first_100)
    # print(mlmzip.replaced_first_100)

    print(text == text2)
    print(len(text))
    print(len(text2))