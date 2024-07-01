import yaml
from services.rays.image_generating import ModelDeployment
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.validator import get_challenge, add_time_penalty
import hashlib
import time, json, os, copy
import requests
from services.offline_rewarding.redis_client import RedisClient
from services.rewarding.cosine_similarity_compare import CosineSimilarityReward

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)
print(MODEL_CONFIG)

class RewardApp():
    def __init__(self, validator):
        self.redis_client: RedisClient = validator.redis_client
        self.reward_stream_name = self.redis_client.reward_stream_name
        self.base_synapse_stream_name = self.redis_client.base_synapse_stream_name

        self.rewarder = CosineSimilarityReward()
        self.validator = validator
        self.validator_response = {}
        self.log_validator_response_engine = "redis"
        if self.log_validator_response_engine == "redis":
            self.redis_key_ttl = 60 * 15
        self.log_validator_response_dir = "log/validator_response"
        if not os.path.exists(self.log_validator_response_dir):
            os.makedirs(self.log_validator_response_dir)

        self.reward_endpoint = "http://213.173.98.5:13300/generate"
        self.current_model = None

        self.total_uids = []
        self.total_rewards = []

    def save_log_validator(self, key, value):
        if self.log_validator_response_engine == "redis":
            self.redis_client.client.set(key, json.dumps(value))
            self.redis_client.client.expire(key, self.redis_key_ttl)
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
        dt = copy.deepcopy(base_synapse)
        dt.pop("message_id", None)
        hash_id =  hashlib.sha256(str(dt).encode('utf-8')).hexdigest()
        return hash_id

    
    def get_challenge_result(self, model_name, base_synapse):
        if model_name in ["GoJourney","DallE"]:
            return None
        req = {
            "model_name": model_name,
            "prompts": [base_synapse],
        }
        response = requests.post(self.reward_endpoint, json = req)
        if response.status_code != 200:
            raise Exception(f"Error in get_challenge_result: {response.text}")
        result = response.json()[0]["image"] # for image models, not implement for text
        return result

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
        success_ids, error_ids = [], []
        for base_synapse in base_synapses:
            raw_message_id = copy.deepcopy(base_synapse["message_id"])
            try:
                hash_id = base_synapse["hash_id"]
                if not self.check_exists_log(hash_id):
                    base_synapse["validator_response"] = self.get_challenge_result(model_name, base_synapse)
                    self.current_model = model_name
                    self.save_log_validator(hash_id, base_synapse)
                else:
                    base_synapse = self.get_log_validator(hash_id)
                success_ids.append(raw_message_id)
            except Exception as ex:
                print(f"[ERROR] generate_response fail: {str(ex)}")
                error_ids.append(raw_message_id)
        return success_ids, error_ids
    
    def group_miner_data_by_model(self, data):
        data_group_by_model = {}
        for d in data:
            base_synapse = d["base_data"]
            model_name = base_synapse["model_name"]
            if model_name not in data_group_by_model:
                data_group_by_model[model_name] = []
            data_group_by_model[model_name].append(d)
        return data_group_by_model

    async def generate_image_response(self, synapse_info, model_name, generate_if_not_exist = False):
        success_ids, not_processed_ids = [], []
        success_synapses = []
        for synapse in synapse_info:
            raw_message_id = copy.deepcopy(synapse["message_id"])
            if len(synapse["valid_uids"]) > 0:
                base_synapse = synapse["base_data"]
                base_synapse["hash_id"] = self.get_base_synapse_hashid(base_synapse)
                if not self.check_exists_log(base_synapse["hash_id"]):
                    if generate_if_not_exist:
                        try:
                            base64_image = self.get_challenge_result(model_name, base_synapse)
                            self.current_model = model_name
                            synapse["validator_response"] = base64_image
                            base_synapse["validator_response"] = base64_image
                            self.save_log_validator(base_synapse["hash_id"], base_synapse)

                            success_ids.append(raw_message_id)
                            success_synapses.append(synapse)
                        except Exception as ex:
                            print(f"Genereate response error: {str(ex)}")
                            not_processed_ids.append(raw_message_id)
                    else:
                        not_processed_ids.append(raw_message_id)
                    
                else:
                    base_synapse = self.get_log_validator(base_synapse["hash_id"])
                    synapse["validator_response"] = base_synapse["validator_response"]

                    success_ids.append(raw_message_id)
                    success_synapses.append(synapse)
        return success_synapses, success_ids, not_processed_ids

    def calculate_rewards(self, data):
        
        total_uids, total_rewards = [], []
        for info in data:
            valid_rewards = []
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
    
    def get_priority_of_model(self):
        """
        Retrieves the processing priority for models. The current model has the highest priority,
        followed by sorting based on the model incentive weight.
        """
        models_with_weights = [(model_name, details["model_incentive_weight"]) for model_name, details in self.validator.nicheimage_catalogue.items()]
        sorted_models = sorted(models_with_weights, key=lambda x: -x[1])
        sorted_model_names = [model for model, weight in sorted_models]

        if self.current_model in sorted_model_names:
            sorted_model_names.remove(self.current_model)
            sorted_model_names.insert(0, self.current_model)
        return sorted_model_names

    async def reward_image_type(self, data, model_name):
        data_with_validator_response, success_ids, not_processed_ids = await self.generate_image_response(data, model_name)    
        total_uids, total_rewards = self.calculate_rewards(data_with_validator_response)
        return {
            "total_uids": total_uids,
            "total_rewards": total_rewards,
            "success_ids": success_ids,
            "not_processed_ids": not_processed_ids
        }

    async def categorize_rewards_type(self, data_group_by_model):
        total_success_ids, total_not_processed_ids = [], []
        for model_name in data_group_by_model:
            reward_url = self.validator.nicheimage_catalogue[model_name]["reward_url"]
            reward_type = self.validator.nicheimage_catalogue[model_name]["reward_type"]
            data = data_group_by_model[model_name]
            reward_uids, rewards, success_ids, not_processed_ids  = [], [], [], []
            if reward_type == 'image':
                image_results =  await self.reward_image_type(data, model_name)
                reward_uids, rewards, success_ids,  not_processed_ids = image_results["total_uids"], image_results["total_rewards"], image_results["success_ids"], image_results["not_processed_ids"]

            elif reward_type == "custom_offline" and callable(reward_url):
                for d in data:
                    d_uids, d_rewards = reward_url(
                        ImageGenerating(**d["base_data"]), [ImageGenerating(**synapse) for synapse in d["all_miner_data"]], d["uids"]
                    )
                    reward_uids.extend(d_uids)
                    rewards.extend(d_rewards)
                success_ids, not_processed_ids = [x["message_id"] for x in data], []
            else:
                print("Reward method not found !!!")
            
            total_success_ids.extend(success_ids)
            total_not_processed_ids.extend(not_processed_ids)
            if len(reward_uids) > 0 :
                reward_uids, rewards =self.scale_reward(reward_uids, rewards)
                print("Reward result: ", reward_uids, rewards)
                self.validator.miner_manager.update_scores(reward_uids, rewards)

                self.total_uids.extend(reward_uids)
                self.total_rewards.extend(rewards)
                print("Total_uids: " ,self.total_uids, f" Total {len(self.total_uids)}", f" Total distinct: {len(list(set(self.total_uids)))}")
                print("Total_rewards: " ,self.total_rewards, f" Total {len(self.total_rewards)}")
        return total_success_ids, total_not_processed_ids

    async def dequeue_reward_message(self):
        async def reward_callback(messages):
            meta = {"count_success" : {}}
            message_ids = [x["id"] for x in messages]
            message_content = [x["content"] for x in messages]

            message_data = []
            for content, id in zip(message_content, message_ids):
                content = json.loads(content["data"])
                content["message_id"] = id
                message_data.append(content)
            data_group_by_model = self.group_miner_data_by_model(message_data)

            total_success_ids, total_not_processed_ids = await self.categorize_rewards_type(data_group_by_model)

            for mess in message_data:
                if mess["message_id"] in total_success_ids:
                    base_synapse = mess["base_data"]
                    model_name = base_synapse["model_name"]
                    if model_name not in meta:
                        meta["count_success"][model_name] = 0
                    meta["count_success"][model_name]+= 1
            return total_success_ids, total_not_processed_ids, meta
        
        await self.redis_client.process_message_from_stream_async(self.reward_stream_name, reward_callback, count=50)

    async def dequeue_base_synapse_message(self):
        async def generate_validator_response(messages):
            meta = {"count_success" : {}}
            message_data = [x["content"] for x in messages]
            message_ids = [x["id"] for x in messages]
            base_synapses = []
            for content, id in zip(message_data, message_ids):
                content = json.loads(content["data"])
                content["message_id"] = id
                base_synapses.append(content)

            data_group_by_model = self.group_synapse_by_model(base_synapses)
            priority_models = self.get_priority_of_model()

            total_success_ids, total_error_ids = [], []
            for model_name in priority_models:
                if model_name not in meta:
                    meta["count_success"][model_name] = 0
                if model_name in data_group_by_model:
                    success_ids, error_ids = await self.generate_response(data_group_by_model[model_name], model_name)
                    total_success_ids.extend(success_ids)
                    total_error_ids.extend(error_ids)
                    meta["count_success"][model_name]+= len(success_ids)
                    break
            
            return total_success_ids, total_error_ids, meta
        
        await self.redis_client.process_message_from_stream_async(self.base_synapse_stream_name, generate_validator_response, count=100)    

if __name__ == "__main__":
    app = RewardApp()

    import asyncio
    asyncio.run(app.dequeue_reward_message())