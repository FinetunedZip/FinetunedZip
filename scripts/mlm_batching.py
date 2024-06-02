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
        num_to_mask = int(token_length * 0.5)  # Calculate the number of tokens to mask based on the desired masking percentage
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

    def mask_batch_tokens(self, tokens):
        # tokens is a batch of sequences, with shape [batch_size, seq_length]
        batch_size, seq_length = tokens.size()
        # Initialize masks for each sequence in the batch
        masks = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=tokens.device)

        # Apply deterministic masks to each sequence in the batch
        for i in range(batch_size):
            individual_mask = self.create_deterministic_masks(seq_length)
            masks[i] = individual_mask

        masked_tokens = tokens.clone()
        # Apply the mask to the token IDs
        masked_tokens[masks] = self.tokenizer.mask_token_id
        return masked_tokens, masks

    def get_predictions(self, masked_tokens):
        masked_tokens = masked_tokens.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(masked_tokens) # auto does batching
        logits = outputs.logits
        return logits

    def get_batch_predictions(self, masked_tokens, attention_masks):
        with torch.no_grad():
            outputs = self.model(masked_tokens, attention_mask=attention_masks)
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
        masked_tokens[masks] = ranks[masks].to(masked_tokens.device)
        # print(masked_tokens)
        return masked_tokens

    def replace_masked_batch_tokens(self, logits, masked_tokens, masks, original_tokens):
        # size of logits is [batch_size, seq_length, vocab_size] and size of masked_tokens is [batch_size, seq_length]
        # we want to store ranks for each token for each sequence in the batch
        # so we need to iterate over the batch dimension and calculate ranks for each sequence

        # return tensor should have shape [batch_size, seq_length]

        # Get the model's predictions for each token
        predictions = torch.argsort(logits, dim=-1, descending=True)

        # Initialize a tensor to store the ranks of each token in the batch
        ranks = torch.full_like(masked_tokens, -1)

        # Iterate over the batch dimension
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                if masks[i][j]:
                    ranks[i][j] = (predictions[i][j] == original_tokens[i][j]).nonzero(as_tuple=True)[0]

        self.ranks.extend(ranks.cpu().numpy())
        masked_tokens[masks] = ranks[masks].to(masked_tokens.device)
        return masked_tokens

    def encode_and_zip(self, text):
        tokens = self.text_to_tokens(text)
        print(len(tokens))
        self.tokens = tokens
        token_chunks = tokens.split(self.CONTEXT_SIZE)
        decoded_tokens_list = []
        map_matrix = []

        token_chunk_batches = [token_chunks[i:i+self.BATCH_SIZE] for i in range(0, len(token_chunks), self.BATCH_SIZE)]
        print("TOTAL BATCHES", len(token_chunk_batches))
        print([len(batch) for batch in token_chunk_batches])

        with tqdm(total=len(token_chunk_batches)) as pbar:
            for batch_index, token_chunk_batch in enumerate(token_chunk_batches):
                # Pad the sequences in the batch to the same length
                max_len = max(chunk.size(0) for chunk in token_chunk_batch)
                # print("MAX LEN", (max_len))

                padded_tokens = torch.stack([torch.nn.functional.pad(chunk, (0, max_len - chunk.size(0)), value=self.tokenizer.pad_token_id) for chunk in token_chunk_batch])
                # print("PADDED TOKENS", padded_tokens.size())  # torch.Size([16, 512])

                # Mask and get predictions
                masked_tokens, masks = self.mask_batch_tokens(padded_tokens)
                # print("MASKED TOKENS", masked_tokens.size())  # torch.Size([16, 512])

                # we need attention masks if there are padded tokens anywhere in the batch
                attention_masks = torch.ones_like(padded_tokens)
                attention_masks[padded_tokens == self.tokenizer.pad_token_id] = 0

                logits = self.get_batch_predictions(masked_tokens, attention_masks=attention_masks)
                # print("LOGITS", logits.size())   # torch.Size([16, 512, 30522])

                decoded_tokens = self.replace_masked_batch_tokens(logits, masked_tokens, masks, padded_tokens)

                # print("DECODED TOKENS", decoded_tokens.size())  # torch.Size([16, 512])

                for i in range(decoded_tokens.size(0)):
                    for j in range(decoded_tokens.size(1)):
                        if masks[i][j] == False and str(decoded_tokens[i][j].item()) in self.frequency_mapping and decoded_tokens[i][j].item() > self.frequency_mapping[str(decoded_tokens[i][j].item())]:
                            decoded_tokens[i][j] = torch.tensor(self.frequency_mapping[str(decoded_tokens[i][j].item())], device=self.device)
                            map_matrix.append(1)
                        else:
                            map_matrix.append(0)

                # frequency_map_tensor = torch.tensor([self.frequency_mapping.get(str(token.item()), token.item()) for token in decoded_tokens.view(-1)], device=self.device).view_as(decoded_tokens)
                # decoded_tokens = torch.where((masks == False) & (decoded_tokens > frequency_map_tensor), frequency_map_tensor, decoded_tokens)

                # # Create map matrix using vectorized operations
                # map_matrix = ((masks == False) & (decoded_tokens > frequency_map_tensor)).int().flatten()

                decoded_tokens_list.extend(decoded_tokens)
                self.mapped_mask.extend(map_matrix)
                map_matrix = []
                pbar.update(1)
                pbar.set_description(f"Encoding Progress: {((batch_index+1)/len(token_chunk_batches))*100:.2f}%")

        # exit(1)

        # Concatenate all decoded tokens into a single tensor
        all_decoded_tokens = torch.cat(decoded_tokens_list)
        map_mask = torch.tensor(self.mapped_mask, dtype=torch.float32, device=self.device)

        # concatenate all_decoded_tokens and map_mask into a single tensor with shape (1, num_tokens * 2)
        zip_package = torch.cat([all_decoded_tokens, map_mask])

        tensor_np = tensor_to_numpy(zip_package)
        print(tensor_np[:5])

        # Compress data using the new codec
        codec = getCodec('optpfor')
        inpComp = np.zeros(tensor_np.size, dtype=np.uint32, order='C')
        compSize = codec.encodeArray(tensor_np, tensor_np.size, inpComp, len(inpComp))

        compressed_array = inpComp[:compSize]
        # Convert compressed data back to bytes for storage
        compressed_bytes = compressed_array.tobytes()
        return compressed_bytes

    def unzip_and_decode(self, zipped_data):
        codec = getCodec('optpfor')
        compSize = len(zipped_data) // 4
        inpComp = np.frombuffer(zipped_data, dtype=np.uint32)
        inpCompDecomp = np.zeros(len(inpComp) * 6, dtype=np.uint32)
        size = codec.decodeArray(inpComp, compSize, inpCompDecomp, len(inpComp) * 6)

        loaded_tensor = numpy_to_tensor(inpCompDecomp, self.device)
        loaded_tensor = loaded_tensor[:size].to(torch.int32)

        # Splitting into original tokens and map_mask
        tensor, map_mask = torch.chunk(loaded_tensor, 2)

        decoded_text_chunks = []
        for i in range(0, tensor.size(0), self.BATCH_SIZE * self.CONTEXT_SIZE):
            end_index = i + self.BATCH_SIZE * self.CONTEXT_SIZE
            batch_tensor = tensor[i:end_index]
            batch_map_mask = map_mask[i:end_index]

            # Reshape to (batch_size, sequence_length)
            batch_tensor = batch_tensor.view(self.BATCH_SIZE, self.CONTEXT_SIZE)
            batch_map_mask = batch_map_mask.view(self.BATCH_SIZE, self.CONTEXT_SIZE)

            # Generate attention masks
            attention_masks = torch.ones_like(batch_tensor)
            attention_masks[batch_tensor == self.tokenizer.pad_token_id] = 0

            mask = self.create_deterministic_masks(self.CONTEXT_SIZE).repeat(self.BATCH_SIZE, 1)

            # Update tensor based on frequency mapping
            for j in range(self.BATCH_SIZE):
                for k in range(self.CONTEXT_SIZE):
                    if batch_map_mask[j][k] == 1:
                        if str(batch_tensor[j][k].item()) in self.reverse_frequency_mapping:
                            batch_tensor[j][k] = torch.tensor(self.reverse_frequency_mapping[str(batch_tensor[j][k].item())], device=self.device)

            masked_tokens = batch_tensor.clone()
            masked_tokens[mask] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_tokens, attention_mask=attention_masks)
            logits = outputs.logits
            predictions = torch.argsort(logits, dim=-1, descending=True)

            # Update tokens based on predictions
            for j in range(self.BATCH_SIZE):
                for k in range(self.CONTEXT_SIZE):
                    if mask[j][k]:
                        batch_tensor[j][k] = predictions[j][k][batch_tensor[j][k]]

            # Decode batch into text
            for j in range(self.BATCH_SIZE):
                decoded_text_chunk = self.tokenizer.decode(batch_tensor[j], skip_special_tokens=True)
                decoded_text_chunks.append(decoded_text_chunk)

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

    def compare_tokens(self, new_tokens):
        # compare the new_tokens to self.tokens
        # print out their lengths and also print out any values at indices where they differ
        if len(new_tokens) != len(self.tokens):
            print("Lengths of the two token lists do not match.")

        for i in range(len(new_tokens)):
            if new_tokens[i] != self.tokens[i]:
                print(f"Tokens at index {i} differ: {self.tokens[i]} vs {new_tokens[i]}")

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

    # print("TEXT 2")
    # print(text2[:100])

    # print(mlmzip.first_100)
    # print(mlmzip.replaced_first_100)

    print(text == text2)
    print(len(text))
    print(len(text2))