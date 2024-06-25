import yaml
from services.rays.image_generating import ModelDeployment
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.validator import get_challenge, add_time_penalty
import hashlib
import time, json, os
from services.offline_rewarding.redis_client import RedisClient
from services.rewarding.cosine_similarity_compare import CosineSimilarityReward

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)
print(MODEL_CONFIG)

class RewardApp():
    def __init__(self, validator):
        
        self.stream_name = "synapse_data"
        self.base_synapse_stream_name = "base_synapse"
        self.redis_client: RedisClient = RedisClient()
        self.rewarder = CosineSimilarityReward()
        self.validator = validator
        self.validator_response = {}
        self.log_validator_response_engine = "redis"
        self.log_validator_response_dir = "log/validator_response"
        if not os.path.exists(self.log_validator_response_dir):
            os.makedirs(self.log_validator_response_dir)

    def save_log_validator(self, key, value):
        if self.log_validator_response_engine == "redis":
            self.redis_client.client.set(key, json.dumps(value))
        else:
            save_path = os.path.join(self.log_validator_response_dir, f'{key}.json')
            with open(save_path , 'w') as f:
                json.dump(value, f, ensure_ascii =False)
    
    def get_log_validator(self, key):
        if self.log_validator_response_engine == "redis":
            data_str = self.redis_client.client.get(key)
            data = json.loads(data_str)
        else:
            save_path = os.path.join(self.log_validator_response_dir, f'{key}.json')
            with open(save_path) as f:
                data = json.load(f)
        return data

    def check_exists_log(self, key):
        if self.log_validator_response_engine == "redis":
            if  self.redis_client.client.exists(key):
                return True
        else:
            save_path = os.path.join(self.log_validator_response_dir, f'{key}.json')
            if os.path.exists(save_path):
                return True
        return False

    def get_base_synapse_hashid(self, base_synapse):
        hash_id =  hashlib.sha256(str(base_synapse).encode('utf-8')).hexdigest()
        return hash_id

    
    def group_synapse_by_model(self, data):
        data_group_by_model = {}
        for base_synapse in data:
            base_synapse["hash_id"] = self.get_base_synapse_hashid(base_synapse)
            model_name = base_synapse["model_name"]
            if model_name not in data_group_by_model:
                data_group_by_model[model_name] = []
            data_group_by_model[model_name].append(base_synapse)

        return data_group_by_model

    async def generate_response(self, base_synapses, model_name):
        model_deployment = ModelDeployment(MODEL_CONFIG[model_name])
        synapse_responses = []
        for base_synapse in base_synapses:
            hash_id = base_synapse["hash_id"]
            if not self.check_exists_log(hash_id):
                synapse_data = ImageGenerating(**base_synapse)
                data = synapse_data.deserialize()
                base64_image = await model_deployment.generate(data)
                base_synapse["validator_response"] = base64_image
                self.save_log_validator(hash_id, base_synapse)
            else:
                base_synapse = self.get_log_validator(hash_id)
            synapse_responses.append(base_synapse)
        del model_deployment
        return synapse_responses
    
    def group_miner_data_by_model(self, data):
        data_group_by_model = {}
        for d in data:
            base_synapse = d["base_data"]
            model_name = base_synapse["model_name"]
            if model_name not in data_group_by_model:
                data_group_by_model[model_name] = []
            data_group_by_model[model_name].append(d)
        return data_group_by_model

    async def generate_image_response(self, synapse_info, model_name):
        model_deployment = ModelDeployment(MODEL_CONFIG[model_name])
        for synapse in synapse_info:
            if len(synapse["valid_uids"]) > 0:
                base_synapse = synapse["base_data"]
                base_synapse["hash_id"] = self.get_base_synapse_hashid(base_synapse)
                if not self.check_exists_log(base_synapse["hash_id"]):
                    synapse_data = ImageGenerating(**base_synapse)
                    data = synapse_data.deserialize()
                    base64_image = await model_deployment.generate(data)
                    synapse["validator_response"] = base64_image
                    base_synapse["validator_response"] = base64_image
                    self.save_log_validator(base_synapse["hash_id"], base_synapse)
                else:
                    base_synapse = self.get_log_validator(base_synapse["hash_id"])
                    synapse["validator_response"] = base_synapse["validator_response"]
        del model_deployment
        return synapse_info

    def calculate_rewards(self, data):
        
        total_uids, total_rewards = [], []
        valid_rewards = []
        for info in data:
            timeout = info["timeout"]
            if len(info["valid_uids"]) > 0:
                validator_response = info.get("validator_response")
                miner_responses = [x["image"] for x in  info["miner_data"]]
                valid_rewards = self.rewarder.get_reward(validator_response, miner_responses)
                valid_rewards = [float(reward) for reward in valid_rewards]
                print(valid_rewards, flush=True)
                process_times = [x["process_time"] for x in info["miner_data"]]
                if timeout > 12:
                    valid_rewards = add_time_penalty(valid_rewards, process_times, 0.4, 64)
                else:
                    valid_rewards = add_time_penalty(valid_rewards, process_times, 0.4, 12)
                valid_rewards = [round(num, 3) for num in valid_rewards]
            uids = info["valid_uids"] + info["invalid_uids"]
            rewards = valid_rewards + [0] * len(info["invalid_uids"])
            
            total_uids.extend(uids)
            total_rewards.extend(rewards)
        
        return total_uids, total_rewards

    # Scale Reward based on Miner Volume
    def scale_reward(self, reward_uids, rewards):     
        for i, uid in enumerate(reward_uids):
            if rewards[i] > 0:
                rewards[i] = rewards[i] * (
                    0.6 + 0.4 * self.validator.miner_manager.all_uids_info[uid]["reward_scale"]
                )
        return reward_uids, rewards
    

    async def reward_image_type(self, data, model_name):
        data_with_validator_response = await self.generate_image_response(data, model_name)    
        total_uids, total_rewards = self.calculate_rewards(data_with_validator_response)
        return total_uids, total_rewards

    def reward_custom_type(self, data, model_name):
        pass
        
    async def categorize_rewards_type(self, data_group_by_model):
        for model_name in data_group_by_model:
            reward_url = self.validator.nicheimage_catalogue[model_name]["reward_url"]
            reward_type = self.validator.nicheimage_catalogue[model_name]["reward_type"]
            data = data_group_by_model[model_name]
            reward_uids, rewards = [], []
            if reward_type == 'image':
                reward_uids, rewards = await self.reward_image_type(data, model_name)
            elif reward_type == "custom" and callable(reward_url):
                for d in data:
                    d_uids, d_rewards = reward_url(
                        ImageGenerating(**d["base_data"]), [ImageGenerating(**synapse) for synapse in d["all_miner_data"]], d["uids"]
                    )
                    reward_uids.extend(d_uids)
                    rewards.extend(d_rewards)
            else:
                print("Reward method not found !!!")
            
            if len(reward_uids) > 0 :
                reward_uids, rewards =self.scale_reward(reward_uids, rewards)
                self.validator.miner_manager.update_scores(reward_uids, rewards)

    async def dequeue_reward_message(self):
        async def reward_callback(message_data):
            message_data = [json.loads(x["data"]) for x in message_data]
            data_group_by_model = self.group_miner_data_by_model(message_data)

            await self.categorize_rewards_type(data_group_by_model)

        
        await self.redis_client.process_message_from_stream_async(self.stream_name, reward_callback)

    async def dequeue_base_synapse_message(self):
        async def generate_validator_response(message_data):
            base_synapses = [json.loads(x["data"]) for x in message_data]
            data_group_by_model = self.group_synapse_by_model(base_synapses)
            for model_name, base_synapses in data_group_by_model.items():
                await self.generate_response(base_synapses, model_name)
            # await self.categorize_rewards_type(data_group_by_model)

        
        await self.redis_client.process_message_from_stream_async(self.base_synapse_stream_name, generate_validator_response)    

if __name__ == "__main__":
    app = RewardApp()

    import asyncio
    asyncio.run(app.dequeue_reward_message())