# Rewarding API for Fixed-Image-Subnet

This API facilitates the re-generation of requests and subsequent verification of miner-validator image pairs utilizing [imagehash](https://github.com/coenm/ImageHash).

## Getting Started
Follow these steps to set up and run the Rewarding API:

1. **Download the Checkpoint**
   - Run the following command to download the necessary checkpoint file:
     ```bash
     bash download_checkpoint.sh
     ```

2. **Build the Docker Image**
   - Create a Docker image named `sd_verifier` by executing:
     ```bash
     docker build -t sd_verifier .
     ```

3. **Run the Docker Container**
   - To run the Docker container and mount the downloaded checkpoint, use the command below:
     ```bash
     docker run --expose 8000 -v ./model.safetensors:/mode.safetensors --gpus all sd_verifier
     ```
   - Ensure that the path to the checkpoint file (`./model.safetensors`) is correct and accessible.

4. **Test API**
    - Go to `http://localhost:8000/docs` to test the API.