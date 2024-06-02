import json
from decimal import Decimal, getcontext
from tqdm import tqdm

# Set the precision high enough for accurate arithmetic coding
getcontext().prec = 80000000  # Increase precision

def calculate_probabilities(sequence):
    frequencies = {}
    print('eefefef')
    for symbol in tqdm(sequence, desc="Calculating Frequencies"):
        if symbol in frequencies:
            frequencies[symbol] += 1
        else:
            frequencies[symbol] = 1

    print('fjfjsjfs')

    total_count = len(sequence)
    probabilities = {}
    for symbol, count in tqdm(frequencies.items(), desc="Calculating Probabilities"):
        probabilities[symbol] = Decimal(count) / Decimal(total_count)
    print('deonnenene')
    return probabilities

def arithmetic_encoding(sequence):
    probabilities = calculate_probabilities(sequence)
    symbols = list(probabilities.keys())
    cumulative_probs = {}
    cumulative = Decimal(0)
    print('hello')
    for symbol in tqdm(symbols, desc="Calculating Cumulative Probabilities"):
        cumulative_probs[symbol] = (cumulative, cumulative + probabilities[symbol])
        cumulative += probabilities[symbol]
    print('ehllo')
    low, high = Decimal(0), Decimal(1)
    for symbol in tqdm(sequence, desc="Encoding Progress"):
        symbol_low, symbol_high = cumulative_probs[symbol]
        interval_range = high - low
        high = low + symbol_high * interval_range
        low = low + symbol_low * interval_range

    return (low + high) / 2, cumulative_probs, probabilities

def save_encoded_data(filename, encoded_value, cumulative_probs, probabilities):
    data = {
        'encoded_value': str(encoded_value),
        'cumulative_probs': {str(k): (str(v[0]), str(v[1])) for k, v in cumulative_probs.items()},
        'probabilities': {str(k): str(v) for k, v in probabilities.items()}
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_encoded_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    encoded_value = Decimal(data['encoded_value'])
    cumulative_probs = {int(k): (Decimal(v[0]), Decimal(v[1])) for k, v in data['cumulative_probs'].items()}
    probabilities = {int(k): Decimal(v) for k, v in data['probabilities'].items()}
    return encoded_value, cumulative_probs, probabilities

def arithmetic_decoding(encoded_value, length, cumulative_probs):
    symbols = list(cumulative_probs.keys())
    sequence = []
    epsilon = Decimal('1e-15000')  # Small value to prevent division by zero

    low, high = Decimal(0), Decimal(1)
    for i in range(length):
        interval_range = high - low
        if interval_range < epsilon:
            interval_range = epsilon
        value = (encoded_value - low) / interval_range
        for symbol in symbols:
            symbol_low, symbol_high = cumulative_probs[symbol]
            if symbol_low <= value < symbol_high:
                sequence.append(symbol)
                high = low + symbol_high * interval_range
                low = low + symbol_low * interval_range
                break

        # Debug statements to trace computation
        print(f"Step {i}:")
        print(f"  low: {low}")
        print(f"  high: {high}")
        print(f"  interval_range: {interval_range}")
        print(f"  value: {value}")
        print(f"  current symbol: {sequence[-1]}")

        if high == low:
            print("High and low are equal. Breaking the loop.")
            break

    return sequence

def encode_to_file(sequence, filename):
    encoded_value, cumulative_probs, probabilities = arithmetic_encoding(sequence)
    save_encoded_data(filename, encoded_value, cumulative_probs, probabilities)
    print(f"Encoded value saved to {filename}")

def decode_from_file(filename, length):
    encoded_value, cumulative_probs, probabilities = load_encoded_data(filename)
    print(f"Loaded encoded value: {encoded_value}")
    print(f"Loaded cumulative probabilities: {cumulative_probs}")
    print(f"Loaded probabilities: {probabilities}")
    decoded_sequence = arithmetic_decoding(encoded_value, length, cumulative_probs)
    return decoded_sequence

def read_integers_from_file(file_path):
    sequence = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                number = int(line.strip())
                sequence.append(number)
            except ValueError:
                # Skip lines that cannot be converted to an integer
                continue
    return sequence

def encode_integers_from_file(input_file, output_file):
    sequence = read_integers_from_file(input_file)
    encode_to_file(sequence, output_file)

# Example usage
input_file = 'zipped/meta-llama-Llama-2-7b-hf-enwik16mb_64_r16_512_16.txt'  # This file should contain one integer per line
output_file = 'encoded_data.json'

# Encode integers from the file and save to output file
encode_integers_from_file(input_file, output_file)

# Decode from output file
sequence = read_integers_from_file(input_file)
decoded_sequence = decode_from_file(output_file, len(sequence))
print("Decoded sequence:", decoded_sequence)
