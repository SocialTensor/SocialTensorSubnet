import json
import zlib
import base64


def compress_dict(data) -> str:
    # Convert dictionary to JSON string
    json_str = json.dumps(data)

    # Compress the JSON string using zlib
    compressed_data = zlib.compress(json_str.encode('utf-8'))

    # Encode the compressed data in base64
    encoded_data = base64.b64encode(compressed_data)

    # Convert the base64 bytes to a string
    compressed_str = encoded_data.decode('utf-8')
    return compressed_str

def decompress_dict(compressed_str) -> dict:
    # Decode the base64 string to bytes
    decoded_data = base64.b64decode(compressed_str)

    # Decompress the zlib-compressed data
    decompressed_data = zlib.decompress(decoded_data)

    # Convert the JSON string back to a dictionary
    original_data = json.loads(decompressed_data.decode('utf-8'))
    return original_data