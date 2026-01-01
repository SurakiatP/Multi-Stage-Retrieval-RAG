import json
import os
import sys

try:
    from toon_format import encode
except ImportError:
    print("Error: 'toon_format' library not found.")
    print("Please install via: pip install git+https://github.com/toon-format/toon-python.git")
    sys.exit(1)

def flatten_data(data):
    """
    Flattens nested JSON structure for TOON tabular format compatibility.
    """
    flattened_list = []
    for item in data:
        new_item = {
            "source": item.get("metadata", {}).get("source", "unknown"),
            "page": item.get("metadata", {}).get("logical_page", "0"),
            "content": item.get("content", "")
        }
        flattened_list.append(new_item)
    return flattened_list

def convert_json_to_toon(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading JSON data from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("Flattening data structure for Tabular format...")
        data = flatten_data(data)

        record_count = len(data)
        print(f"Processing {record_count} records.")

        print("Encoding to TOON format...")
        toon_output = encode(data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(toon_output)

        print(f"Success! Saved TOON file to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    INPUT_FILE = os.path.join(BASE_DIR, "ingested_data", "ingested_documents.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "ingested_data", "ingested_documents.toon")
    
    convert_json_to_toon(INPUT_FILE, OUTPUT_FILE)