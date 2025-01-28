import timm
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageChops
import torch
from typing import List
from generation_models.utils import base64_to_pil_image
import imagehash
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class CosineSimilarityReward(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
        threshold=0.9,
        device=None,
    ):
        super(CosineSimilarityReward, self).__init__()
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.threshold = threshold
        self.model, self.transforms = self.get_model(model_name)

    def get_model(self, model_name):
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()
        model.to(self.device)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    @torch.inference_mode()
    def forward(
        self, validator_image: Image.Image, miner_image: Image.Image, binary=True
    ) -> float:
        validator_vec = self.model(
            self.transforms(validator_image).unsqueeze(0).to(self.device)
        )
        image_vec = self.model(
            self.transforms(miner_image).unsqueeze(0).to(self.device)
        )
        cosine_similarity = F.cosine_similarity(validator_vec, image_vec)
        if binary:
            if cosine_similarity.item() > self.threshold:
                reward = 1.0
            elif cosine_similarity.item() > 0.4:
                reward = (cosine_similarity.item() + (1 - self.threshold)) ** 3
            else:
                reward = 0.0

            print(f"Sim: {cosine_similarity.item()} -> reward: {reward}")
            return reward

        return float(cosine_similarity.item())

    def get_reward(
        self,
        validator_image: Image.Image,
        batched_miner_images: List[str],
        pipeline_type: str,
    ) -> List[float]:
        rewards = []
        if not isinstance(validator_image, Image.Image):
            validator_image = base64_to_pil_image(validator_image)
        for miner_image in batched_miner_images:
            if not miner_image:
                reward = False
            else:
                try:
                    if not isinstance(miner_image, Image.Image):
                        miner_image = base64_to_pil_image(miner_image)
                except Exception:
                    print("Corrupted miner image", flush=True)
                    reward = False
                    rewards.append(reward)
                    continue
                nsfw_check = self.nsfw_filter(validator_image, miner_image)
                if nsfw_check:
                    reward = -5 if nsfw_check == 2 else 0
                else:
                    if pipeline_type == "upscale":
                        reward = self.calculate_reward_upscale(
                            miner_image, validator_image
                        )
                    else:
                        reward = self.matching_image(miner_image, validator_image)

            rewards.append(reward)
        return rewards

    def get_black_hash(self, H, W) -> str:
        image = Image.new("RGB", (W, H), color="black")
        return str(imagehash.average_hash(image, hash_size=8))

    def matching_image(
        self, miner_image: Image.Image, validator_image: Image.Image
    ) -> bool:
        cosine_similarity_score = self.forward(validator_image, miner_image)
        print("Cosine Similarity Score:", cosine_similarity_score, flush=True)
        return cosine_similarity_score

    def matching_image_2(
        self, miner_image: Image.Image, validator_image: Image.Image
    ) -> bool:
        """Crop image regions with high similarity to calculate cosine similarity"""
        W, H = validator_image.size
        assert validator_image.size == miner_image.size, "Validator and miner image size must be the same"
        nums_crop = 10
        random_crop_size = 192
        
        # Get similarity map
        normalized_diff = self.calculate_diff(miner_image, validator_image)
        
        # Convert to numpy array and find regions with highest similarity
        similarity_map = np.array(normalized_diff)
        score_list = []
        
        # Use sliding window to find regions with highest average similarity
        best_regions = []
        for y in range(0, H - random_crop_size, random_crop_size // 2):
            for x in range(0, W - random_crop_size, random_crop_size // 2):
                region_similarity = np.mean(similarity_map[y:y+random_crop_size, x:x+random_crop_size])
                best_regions.append((x, y, region_similarity))
        
        # Sort regions by similarity and take top N
        best_regions.sort(key=lambda x: x[2], reverse=True)
        best_regions = best_regions[:nums_crop]
        
        # Calculate cosine similarity for best regions
        for x, y, _ in best_regions:
            _validator_image = validator_image.crop((x, y, x + random_crop_size, y + random_crop_size))
            _miner_image = miner_image.crop((x, y, x + random_crop_size, y + random_crop_size))
            cosine_similarity_score = self.forward(_validator_image, _miner_image, binary=False)
            score_list.append(cosine_similarity_score)

        return sum(score_list) / nums_crop
    
    def calculate_diff(img1, img2):
        """
        Compare two PIL images, return their similarity regions.

        Parameters:
            img1 (PIL.Image): The first image.
            img2 (PIL.Image): The second image.
        """
        # Ensure the images are the same size
        if img1.size != img2.size:
            raise ValueError("Images must be the same size for comparison.")

        # Convert images to grayscale
        img1_gray = img1.convert("L")
        img2_gray = img2.convert("L")

        # Compute the absolute difference
        diff = ImageChops.difference(img1_gray, img2_gray)

        # Normalize the difference for visualization
        diff_array = np.array(diff, dtype=np.float32)
        normalized_diff = (255 - (diff_array / diff_array.max()) * 255).astype(np.uint8)

        return normalized_diff

    def nsfw_filter(
        self, validator_image: Image.Image, miner_image: Image.Image
    ) -> bool:
        W, H = validator_image.size
        validator_hash = imagehash.average_hash(validator_image, hash_size=6)
        miner_hash = imagehash.average_hash(miner_image, hash_size=6)
        validator_hash = str(validator_hash)
        miner_hash = str(miner_hash)
        black_hash = self.get_black_hash(H, W)
        if validator_hash != black_hash and miner_hash == black_hash:
            return 1
        if validator_hash == black_hash and miner_hash != black_hash:
            return 2
        return 0

    def calculate_reward_upscale(
        self,
        validator_image: Image.Image,
        miner_image: Image.Image,
        psnr_threshold=30,
        ssim_threshold=0.9,
    ):
        validator_array = np.array(validator_image.convert("L"))
        miner_array = np.array(miner_image.convert("L"))
        psnr_value = psnr(validator_array, miner_array)
        ssim_value, _ = ssim(validator_array, miner_array, full=True)

        if psnr_value >= psnr_threshold and ssim_value >= ssim_threshold:
            reward = 1.0
        elif psnr_value < psnr_threshold and ssim_value < ssim_threshold:
            reward = 0.0
        else:
            # Partial reward based on quality
            psnr_penalty = min(psnr_value / psnr_threshold, 1.0)
            ssim_penalty = min(ssim_value / ssim_threshold, 1.0)
            penalty_factor = 0.6
            reward = penalty_factor * (psnr_penalty + ssim_penalty) / 2
        print("calculate_reward_upscale: ", psnr_value, ssim_value, reward)
        return reward
