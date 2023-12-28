import os
from tqdm import tqdm


# Function to send a single request
def send_request(prompt, model_name):
    command = """
curl -X 'POST' \
  'http://localhost:10003/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "an image of",
  "model_name": "RealisticVision"
}'
"""
    os.system(command)


# Main loop to send 50 requests
for i in tqdm(range(50), total=50):
    send_request("an image of", "RealisticVision")

# Running the script
if __name__ == "__main__":
    send_request()
