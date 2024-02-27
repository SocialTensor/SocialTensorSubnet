# Prompt Generation API for Fixed-Image-Subnet

This API utilizes the [MagicPrompt](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) to generate prompts with a specified seed.

## Getting Started

### Installation Steps

1. **Build the Docker Image**
   - Use the following command to create a Docker image named `prompt_gen`:
     ```bash
     docker build -t prompt_gen .
     ```

2. **Run the Docker Container**
   - Execute the command below to run the Docker container. This command also mounts the downloaded checkpoint and exposes the appropriate port:
     ```bash
     docker run -p 8001:8001 --gpus all prompt_gen
     ```

3. **Test the API**
   - Access the Swagger UI to test the API by navigating to:
     ```
     http://localhost:8001/docs
     ```
     Here, you can send requests and view responses directly from the browser.
