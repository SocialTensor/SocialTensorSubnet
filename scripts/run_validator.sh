python neurons/validator/validator.py \
--netuid 1 \
--subtensor.chain_endpoint ws://20.243.203.20:9946 \
--wallet.name validator --wallet.hotkey default \
--proxy.port 8080 \
--proxy.public_ip http://localhost \
--proxy.market_registering_url http://localhost:10003/get_credentials \
--reward_endpoint http://localhost:10002/verify \
--prompt_generating_endpoint http://localhost:10001/prompt_generate \
--logging.debug \