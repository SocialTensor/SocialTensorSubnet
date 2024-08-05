# Models Description
Welcome to the technical documentation for models used in NicheImage. Below, we provide a detailed description of models available in NicheImage
**Miner feels free to customize the deployment as long as the output is still the same!**
## FluxSchnell
![Flux Samples](/assets/images/flux-1.png)
The FluxSchnell model boasts an impressive 12 billion parameters, making it a powerful tool for advanced image generation. It is designed to run on GPUs with 24GB of VRAM.

### Model Architecture
FluxSchnell comprises two major components: the Text Encoder and the Denoise Model.

#### Text Encoder
- **Model**: T5-XXL
- **VRAM Usage**: Approximately 8GB
- **Performance**: Despite its size, the T5-XXL runs efficiently, providing rapid text encoding.

#### Denoise Model
- **Model**: Flow Matching DiT
- **VRAM Usage**: Approximately 23GB
- **Efficiency**: This model requires only 4 denoise steps, making it highly efficient in terms of processing speed.

### VRAM Optimization
At first glance, the combined VRAM usage of the Text Encoder and the Denoise Model exceeds 24GB. However, with strategic memory management, FluxSchnell can operate within the 24GB limit. By offloading components to the CPU when not in use, we can optimize memory usage. Below is a pseudocode example illustrating this process:

```python
def inference(text_encoder, denoise_model, prompt):
    text_encoder.to("cuda")
    embeddings = text_encoder(prompt)
    text_encoder.to("cpu")

    denoise_model.to("cuda")
    output = denoise_model(embeddings)
    denoise_model.to("cpu")
```

This approach ensures that FluxSchnell runs efficiently on a 24GB VRAM GPU, leveraging both the Text Encoder and Denoise Model to their fullest potential without exceeding memory constraints.


## Kolors: Text To Image, ControlNet

![Kolors Samples](/assets/images/kolors.png)

Kolors is a large-scale text-to-image generation model based on latent diffusion, developed by the Kuaishou Kolors team. Trained on billions of text-image pairs, Kolors exhibits significant advantages over both open-source and proprietary models in visual quality, complex semantic accuracy, and text rendering for both Chinese and English characters. Furthermore, Kolors supports both Chinese and English inputs, demonstrating strong performance in understanding and generating Chinese-specific content.

### Model Architecture

Kolors = SDXL + ChatGLM Text Encoder
By default setup, the text encoder is quatized.
### Pipelines
Kolors supports 3 pipelines: Text To Image, ControlNet (Depth, Canny Edge) and IP Adapter.