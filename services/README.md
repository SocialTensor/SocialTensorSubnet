
**Optional** Validator gets challenge prompt and image to make synthentic request from our endpoint as default, but you can setup your own server
1. Start prompt generating endpoint
```
pm2 start python --name "challenge_prompt" -- -m services.challenge_generating.app --port 11001 --disable_secure
```
2. Start image generating endpoint
```
pm2 start python --name "challenge_image" -- -m services.challenge_generating.app --port 11002 --disable_secure
```
**Optional** Validators gets reward from our endpoint as default, which uses reproducing and hash matching mechanism, but you can setup your own server

**Eg.**
- `AnimeV3`
```
pm2 start python --name "reward_AnimeV3" -- -m services.rewarding.app --port 12001 --model_name AnimeV3 --disable_secure
```
- `RealititesEdgeXL`
```
pm2 start python --name "reward_REXL" -- -m services.rewarding.app --port 12002 --model_name RealitiesEdgeXL --disable_secure
```
- `RealisticVision`
```
pm2 start python --name "reward_RV" -- -m services.rewarding.app --port 12003 --model_name RealisticVision --disable_secure
```
- `DreamShaper`
```
pm2 start python --name "reward_DreamShaper" -- -m services.rewarding.app --port 12004 --model_name DreamShaper --disable_secure
```
**Run validator with your own endpoints**
_replace localhost with another ip if you don't host locally_
```
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> \ # Optional, pass only if you want allow queries through your validator and get paid
--challenge.prompt http://localhost:11001 \ 
--challenge.image http://localhost:11002 \
--reward_url.AnimeV3 http://localhost:12001 \
--reward_url.RealitiesEdgeXL http://localhost:12002 \
--reward_url.RealisticVision http://localhost:12003 \
--reward_url.DreamShaper http://localhost:12004 \
