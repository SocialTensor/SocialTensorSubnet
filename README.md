# Stable Diffusion Subnet

## Getting Started

### Prequisites
Before proceeding further, make sure that you have installed Bittensor. See the below instructions:

- [Install `bittensor`](https://github.com/opentensor/bittensor#install).

After installing `bittensor`, proceed as below:

### 1. Install Substrate dependencies

Begin by installing the required dependencies for running a Substrate node.

Update your system packages:

```bash
sudo apt update 
```

Install additional required libraries and tools

```bash
sudo apt install --assume-yes make build-essential git clang curl libssl-dev llvm libudev-dev protobuf-compiler
```
### 2. Install Rust and Cargo

Rust is the programming language used in Substrate development. Cargo is Rust package manager.

Install rust and cargo:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Update your shell's source to include Cargo's path:

```bash
source "$HOME/.cargo/env"
```

## 3. Clone the subtensor repository

This step fetches the subtensor codebase to your local machine.

```bash
git clone https://github.com/toilaluan/subtensor.git
```

## 4. Setup & Run Subtensor server
```bash
cd subtensor
./scripts/init.sh
./scripts/localnet.sh
```
**NOTE**: Watch for any build or initialization outputs in this step. If you are building the project for the first time, this step will take a while to finish building, depending on your hardware.

## 5. Set up wallets
You will need wallets for the different roles, i.e., subnet owner, subnet validator and subnet miner, in the subnet. 

- The owner wallet creates and controls the subnet. 
- The validator and miner will be registered to the subnet created by the owner. This ensures that the validator and miner can run the respective validator and miner scripts.

Create a coldkey for the owner role:

```bash
btcli wallet new_coldkey --wallet.name owner
```

Set up the miner's wallets:

```bash
btcli wallet new_coldkey --wallet.name miner
```

```bash
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

Set up the validator's wallets:

```bash
btcli wallet new_coldkey --wallet.name validator
```
```bash
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### 6. Mint tokens from faucet

You will need tokens to initialize the intentive mechanism on the chain as well as for registering the subnet. 

- Owner need > τ1000 for create subnet
- Validator & Miner need > τ0 for registering to subnet

Run the following commands to mint faucet tokens for the owner and for the validator.

Mint faucet tokens for the owner:
```bash
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

You will see:
```bash
Run Faucet ?                                                                                                                             
 coldkey:    *****                                                                            
 network:    local [y/n]: y                                                                                                              
Enter password to unlock key:                                                                                                            
Balance: τ0.000000000 ➡ τ100.000000000                                                                                                   
Balance: τ100.000000000 ➡ τ200.000000000                                                                                                 
Balance: τ200.000000000 ➡ τ300.000000000

Run Faucet ?                                                                                                                             
 coldkey:    *****                                                                            
 network:    local [y/n]: y                                                                                                              
Enter password to unlock key:                                                                                                            
Balance: τ400.000000000 ➡ τ500.000000000                                                                                                 
Balance: τ500.000000000 ➡ τ600.000000000                                                                                                 
Balance: τ600.000000000 ➡ τ700.000000000

Run Faucet ?                                                                                                                  [1759/3666]
 coldkey:    *****
 network:    local [y/n]: y
Enter password to unlock key: 
Balance: τ700.000000000 ➡ τ800.000000000
Balance: τ800.000000000 ➡ τ900.000000000
Balance: τ900.000000000 ➡ τ1,000.000000000
```

Mint tokens for the validator:
```bash
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

You will see:
```bash
Run Faucet ?                                                                                                                             
 coldkey:    *****                                                                            
 network:    local [y/n]: y                                                                                                              
Enter password to unlock key:                                                                                                            
Balance: τ0.000000000 ➡ τ100.000000000                                                                                                   
Balance: τ100.000000000 ➡ τ200.000000000                                                                                                 
Balance: τ200.000000000 ➡ τ300.000000000
```

Mint tokens for the miner:
```bash
btcli wallet faucet --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

You will see:
```bash
Run Faucet ?                                                                                                                             
 coldkey:    *****                                                                            
 network:    local [y/n]: y                                                                                                              
Enter password to unlock key:                                                                                                            
Balance: τ0.000000000 ➡ τ100.000000000                                                                                                   
Balance: τ100.000000000 ➡ τ200.000000000                                                                                                 
Balance: τ200.000000000 ➡ τ300.000000000
```

## 7. Create a subnet

The below commands establish a new subnet on the local chain. The cost will be exactly τ1000.000000000 for the first subnet you create.

```bash
btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

You will see:

```bash
>> Your balance is: τ1200.000000000
>> Do you want to register a subnet for τ1000.000000000? [y/n]: 
>> Enter password to unlock key: [YOUR_PASSWORD]
>> ✅ Registered subnetwork with netuid: 1
```

**NOTE**: The local chain will now have a default `netuid` of 1. The second registration will create a `netuid` 2 and so on, until you reach the subnet limit of 8. If you register more than 8 subnets, then a subnet with the least staked TAO will be replaced by the 9th subnet you register.

## 8. Register keys

Register your subnet validator and subnet miner on the subnet. This gives your two keys unique slots on the subnet. The subnet has a current limit of 128 slots.

Register the subnet miner:

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
```

Follow the below prompts:

```bash
Your balance is: τ300.000000000                                                                                                          
The cost to register by recycle is τ1.000000000                                                                                          
Do you want to continue? [y/n] (n): y                                                                                                    
Enter password to unlock key:                                                                                                            
Recycle τ1.000000000 to register on subnet:1? [y/n]: y                                                                                   
📡 Checking Balance...                                                                                                                   
Balance:                                                                                                                                 
  τ300.000000000 ➡ τ299.000000000                                                                                                        
✅ Registered
```

Register the subnet validator:

```bash

btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
```

Follow the below prompts:

```
Your balance is: τ300.000000000                                                                                                          
The cost to register by recycle is τ1.000000000                                                                                          
Do you want to continue? [y/n] (n): y                                                                                                    
Enter password to unlock key:                                                                                                            
Recycle τ1.000000000 to register on subnet:1? [y/n]: y                                                                                   
📡 Checking Balance...                                                                                                                   
Balance:                                                                                                                                 
  τ300.000000000 ➡ τ299.000000000                                                                                                        
✅ Registered
```

## 11. Add stake 

This step bootstraps the incentives on your new subnet by adding stake into its incentive mechanism.

```bash
btcli stake add --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
```

Follow the below prompts:

```bash
Stake all Tao from account: 'validator'? [y/n]: y
2023-12-05 03:35:09.646 |       INFO       | Connected to local network and ws://127.0.0.1:9946.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 295.12it/s]
Do you want to stake to the following keys from validator:
    - default:*****: All
 [y/n]: y
Enter password to unlock key: 
Do you want to stake:
  amount: τ298.999999000
  to: default [y/n]: y
✅ Finalized
Balance:
  τ299.000000000 ➡ τ0.000001000
Stake:
  τ0.000000000 ➡ τ298.999999000
```

