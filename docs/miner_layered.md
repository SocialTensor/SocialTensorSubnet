# Setup Layered Miner

## 1. Setup Layer 0
Set up the Axon to connect to the Bittensor network, allowing validators to query Layer 1.

```bash
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid 23 \
--wallet.name <wallet_name> \
--wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \ # default is finney
--axon.port <your_public_port> \
--miner.is_layer_zero \
--miner.layer_one_ip <layer_one_ip> \ # change your layer one IP, default is 0.0.0.0
--miner.layer_one_port <layer_one_port> # change your layer one port, default is 8091
```

## 2. Setup Layer 1
Set up the miner Axon that only validators can access.

```bash
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid 23 \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \ # default is finney
--generate_endpoint http://localhost:10006/generate \ # change if you use different port or host
--info_endpoint http://localhost:10006/info \ # change if you use different port or host
--axon.port <layer-one-port> \
--miner.total_volume <your-generation-volume> \ # default is 40. Change based on your model timeout value and GPU capacity
--miner.max_concurrent_requests <your-max-concurrent-requests> \ # default is 4. Change based on your model timeout value and GPU capacity
--miner.is_layer_one \
--miner.disable_serve_axon
```

