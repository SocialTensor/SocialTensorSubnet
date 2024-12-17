from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import argparse
from typing import Optional
import uvicorn
import random
import openai
from prometheus_fastapi_instrumentator import Instrumentator


class Prompt(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    seed: Optional[int] = 0
    max_length: Optional[int] = 77
    model_name: Optional[str] = "OpenGeneral"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11277)
    parser.add_argument("--netuid", type=str, default=23)
    parser.add_argument("--min_stake", type=int, default=100)
    parser.add_argument(
        "--chain_endpoint",
        type=str,
        default="finney",
    )
    parser.add_argument("--disable_secure", action="store_true", default=False)
    args = parser.parse_args()
    return args


class ChallengePrompt:
    def __init__(self):
        self.app = FastAPI(title="Challenge Prompt")
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.open_category_prefixes = {
            "OpenGeneral": [
                "an image of",
                "a landscape image of",
                "a painting of",
                "a drawing of",
                "a portrait photo of",
                "an illustration of",
                "a animated image of",
                "a sketch of",
            ],

            ### Digital Art Category
            "OpenDigitalArt": [
                "an digital art of",
                "a digital painting of",
                "a digital drawing of",
                "a digital illustration of",
                "a digital image of",
                "a digital sketch of",
            ],
            "OpenDigitalArtAnime": [
                "an anime-style illustration of",
                "a manga drawing of",
                "a vibrant anime character",
                "a scene from an anime world featuring",
                "a chibi-style drawing of",
            ],
            "OpenDigitalArtMinimalist": [
                "a minimalist illustration of",
                "a simple and clean digital design of",
                "a minimalistic art piece featuring",
                "a geometric abstract artwork of",
                "a minimal line art drawing of",
            ],
            "OpenDigitalArtPixelArt": [
                "a pixel art of",
                "a retro pixelated scene of",
                "a pixel character sprite of",
                "a classic 8-bit artwork of",
            ],

            ### Traditional Art Category
            "OpenTraditionalArt": [
                "a traditional artwork of",
                "a hand-drawn piece of art featuring",
                "a beautiful traditional illustration of",
                "a classic art piece depicting",
                "a traditional depiction of",
                "a detailed handmade artwork of",
                "an artistic creation featuring",
                "a timeless traditional art style of",
            ],
            "OpenTraditionalArtPainting": [
                "a traditional oil painting of",
                "a watercolor painting of",
                "an acrylic painting of",
                "a still life painting of",
                "a historical painting illustrating",
                "an impressionist painting of",
                "a renaissance-style painting of",
            ],
            "OpenTraditionalArtSketch": [
                "a pencil sketch of",
                "a charcoal sketch of",
                "a quick hand-drawn sketch of",
                "a detailed line sketch of",
                "a black-and-white sketch of",
                "a rough draft sketch of",
                "an artistic concept sketch of",
                "a traditional hand-drawn sketch featuring",
            ],
            "OpenTraditionalArtComic": [
                "a traditional comic book illustration of",
                "a hand-drawn comic panel featuring",
                "a vintage comic-style drawing of",
                "a dynamic comic scene depicting",
                "a retro comic strip showing",
                "a hand-inked comic drawing of",
                "a comic book cover art featuring",
            ],  
            
            ### Landscape Category
            "OpenLandscape": [
                "an landscape image of",
                "a landscape photo of",
                "a landscape view of",
            ],
            "OpenLandscapeNature": [
                "a peaceful forest landscape with",
                "a serene mountain view with",
                "a beautiful river",
                "a dense forest with",
                "a vast desert with",
                "a lush landscape with rolling hills and",
                "a peaceful lake surrounded by",
                "a lush valley with",
                "a small tropical islandy",
                "a smoking volcano with",
            ],
            "OpenLandscapeCity": [
                "a bustling cityscape",
                "a vibrant urban landscape",
                "a dynamic view of the city",
                "an urban scene filled with life",
                "a city view full of buildings",
                "a modern city with",
                "a street view in an urban area",
                "a panoramic view of a city",
            ],
            "OpenLandscapeAnimal": [
                "a scene featuring animals",
                "a majestic animal in the wild",
                "a wildlife photo of",
                "a peaceful scene of animals grazing",                
                "a natural setting with wildlife",
                "animals in their natural habitat",
                "a tranquil scene with animals",
                "a landscape with roaming animals",
                "a picturesque scene of wildlife",
            ],

            ### People Category
            "OpenRealisticPeople": [
                "a realistic photo of people",
                "a candid shot of a person",
                "a group of people in a lively setting",
                "a photo capturing human emotions",
                "a casual photo of someone",
            ],
            "OpenPeoplePortrait": [
                "a close-up portrait photo of",
                "a detailed portrait shot of",
                "a studio portrait with soft lighting",
                "a black and white portrait of",
                "a professional headshot of",
            ],
            "OpenPeopleLifestyle": [
                "a lifestyle photo of",
                "a candid lifestyle shot of someone",
                "a person enjoying daily life activities",
                "a photo of someone relaxing",
                "a shot of people in their home environment",
                "a photo of someone enjoying",
                "a person relaxing in their cozy home",
                "a lifestyle photo of someone reading",
                "a photo of a person hiking in the mountains",
                "a candid shot of someone cooking in the kitchen",
                "a person spending time with their pet outdoors",
                "a family enjoying a picnic in the park",
            ],
            "OpenPeopleFashion": [
                "a high-fashion photo of",
                "a fashion editorial shot of",
                "a glamorous runway photo of",
                "a high-fashion editorial photo of",
                "a model wearing",
                "a model showcasing",
                "a glamorous photo of a modelg",
            ]
        }

        self.vllm_client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1")
        Instrumentator().instrument(self.app).expose(self.app)
        
    async def __call__(
        self,
        payload: Prompt,
    ):
        model_name = payload.model_name
        prompt = payload.prompt
        prompts = self.open_category_prefixes[model_name]
        init_prompt = random.choice(prompts)
        prompt = f"<|endoftext|> {init_prompt}"

        output = await self.vllm_client.completions.create(
            model="toilaluan/Image-Caption-Completion-Long",
            prompt=prompt,
            max_tokens=125,
            temperature=1.0,
            top_p=1,
            n=1,
            presence_penalty=0,
        )
        completed_prompt = output.choices[0].text.strip()
        completed_prompt = completed_prompt.replace("\n", " ")
        completed_prompt = completed_prompt.split(".")[:-1]
        completed_prompt = ".".join(completed_prompt) + "."

        return {"prompt": init_prompt + " " + completed_prompt.strip()}


if __name__ == "__main__":
    args = get_args()
    print("Args: ", args)
    app = ChallengePrompt()
    uvicorn.run(app.app, host="0.0.0.0", port=args.port)
