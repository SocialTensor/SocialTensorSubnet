## Optional - Run Prompting API and Reward API as a Validator

1. [Validator] Prompting API 
```bash
python dependency_modules/prompt_generating/app.py --port <port>
```
2. [Validator] Rewarding API
```bash
python dependency_modules/rewarding/app.py --port <port> --model_name <model_name>
```