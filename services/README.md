## Optional - Run Prompting API and Reward API as a Validator

1. [Validator] Prompting API 
```bash
python services/prompt_generating/app.py --port <port>
```
2. [Validator] Rewarding API
```bash
python services/rewarding/app.py --port <port> --model_name <model_name>
```