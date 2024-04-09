**Prequisite for vLLM Category**

Find the model name and repo_id for the model you want to mine with
| Model name | repo_id |
|------------|---------|
| Gemma7B | `google/gemma-7b-it` |

To start mining with this model, follow these steps:
1. Create a new Python environment for `vLLM`:
```bash
python -m venv vllm
source vllm/bin/activate
pip install vllm
```
2. Start the API server with your Hugging Face token (ensure access to the model repo at https://huggingface.co/repo_id):
```bash
HF_TOKEN=<your-huggingface-token> python -m vllm.entrypoints.openai.api_server --model repo_id
```