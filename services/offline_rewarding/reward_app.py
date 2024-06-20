import yaml
from services.rays.image_generating import ModelDeployment
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.validator import get_challenge, add_time_penalty
import pickle
import time, json
from services.offline_rewarding.redis_client import RedisClient
from services.rewarding.cosine_similarity_compare import CosineSimilarityReward

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)
print(MODEL_CONFIG)

class RewardApp():
    def __init__(self, validator):
        
        self.stream_name = "synapse_data"
        self.redis_client: RedisClient = RedisClient()
        self.rewarder = CosineSimilarityReward()
        self.prefetch_count = 3
        self.validator = validator

    def group_synapse_by_model(self, data):
        data_group_by_model = {}
        for d in data:
            base_synapse = d["base_data"]
            model_name = base_synapse["model_name"]
            if model_name not in data_group_by_model:
                data_group_by_model[model_name] = []
            data_group_by_model[model_name].append(d)
        return data_group_by_model

    async def generate_response(self, data_group_by_model):
        for model_name in data_group_by_model:
            data =  data_group_by_model[model_name]
            model_deployment = ModelDeployment(MODEL_CONFIG[model_name])
            for synapse in data_group_by_model[model_name]:
                base_synapse = synapse["base_data"]
                synapse_data = ImageGenerating(**base_synapse)
                data = synapse_data.deserialize()
                base64_image = await model_deployment.generate(data)
                synapse["validator_response"] = base64_image
                del model_deployment
            return data_group_by_model

    def calculate_rewards(self, data_with_validator_response):
        
        total_uids, total_rewards = [], []
        for model_name in data_with_validator_response:
            for info in data_with_validator_response[model_name]:
                all_uids_info = self.validator.miner_manager.all_uids_info
                timeout = info["timeout"]
                validator_response = info["validator_response"]
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
                for i, uid in enumerate(info["valid_uids"]):
                    if valid_rewards[i] > 0:
                        valid_rewards[i] = valid_rewards[i] * (
                            0.6 + 0.4 * all_uids_info[uid]["reward_scale"]
                        )
                rewards = valid_rewards + [0] * len(info["invalid_uids"])
                
                total_uids.extend(uids)
                total_rewards.extend(rewards)
                self.validator.miner_manager.update_scores(total_uids, total_rewards)

    async def dequeue_message(self):
        async def reward_callback(message_data):
            message_data = [json.loads(x["data"]) for x in message_data]
            data_group_by_model = self.group_synapse_by_model(message_data)
            data_with_validator_response = await self.generate_response(data_group_by_model)
            self.calculate_rewards(data_with_validator_response)
        
        await self.redis_client.process_message_from_stream_async(self.stream_name, reward_callback)
                

if __name__ == "__main__":
    app = RewardApp()

    import asyncio
    asyncio.run(app.dequeue_message())