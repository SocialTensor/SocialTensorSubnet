import typesense
import os
import random
from transformers import AutoTokenizer
from cleantext import clean


TYPESENSE_API_KEY = os.environ.get("TYPESENSE_API_KEY")
TYPESENSE_HOST = os.environ.get("TYPESENSE_HOST")
TYPESENSE_PORT = os.environ.get("TYPESENSE_PORT")


class TwitterPrompt:
    def __init__(self, max_tokens=1024):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        self.max_tokens = max_tokens
        self.client = typesense.Client(
            {
                "api_key": TYPESENSE_API_KEY,
                "nodes": [
                    {"host": TYPESENSE_HOST, "port": TYPESENSE_PORT, "protocol": "http"}
                ],
                "connection_timeout_seconds": 12,
            }
        )
        self.total_documents = {}

    def get_text_total_tokens(self, text):
        ids = self.tokenizer(text, return_tensors="pt")
        return len(ids["input_ids"][0])

    def __call__(self):
        tweets = self.get_tweets()
        prompt = self.get_prompt(tweets)
        return prompt

    def get_tweets(self, mode="time_cluster", **kwargs):
        if mode == "time_cluster":
            query = {
                "q": "*",
                "query_by": "text",
                "sort_by": "timestamp:desc",
                "filter_by": "imagesLength:=0 && likes:>0",
                "per_page": 1,
            }
            if mode in self.total_documents:
                num_documents = self.total_documents[mode]
            else:
                num_documents = self.get_total_tweets(query)
                self.total_documents[mode] = num_documents
            offset = random.randint(0, num_documents)
            query["per_page"] = 25
            query["offset"] = offset
            print(query)
            response = self.client.collections["posts"].documents.search(query)
            tweets = response["hits"]
            # sort tweets by likes
            tweets = [tweet["document"] for tweet in tweets]
            tweets = sorted(tweets, key=lambda x: x["likes"], reverse=True)
            texts = [tweet["text"] for tweet in tweets]
            texts = [self.clean_tweet(tweet) for tweet in texts]
            users = [tweet.get("username", "N/A") for tweet in tweets]
            timestamps = [
                tweet.get("timestamp", "N/A").split(" ")[0] for tweet in tweets
            ]
            likes = [tweet.get("likes", 1) for tweet in tweets]
            hashtags = [tweet.get("hashtags", "[]") for tweet in tweets]
            tweets_prompts = []
            for text, user, timestamp, like, hashtag in zip(
                texts, users, timestamps, likes, hashtags
            ):
                tweets_prompts.append(
                    f"- Tweet by user {user} on {timestamp} with {like} likes: {text}\n"
                )
            tweets_prompt = "\n".join(tweets_prompts)
            while (
                self.get_text_total_tokens(tweets_prompt) > self.max_tokens
                and tweets_prompts
            ):
                tweets_prompts.pop()
                tweets_prompt = "\n".join(tweets_prompts)
            print(f"The prompt has {self.get_text_total_tokens(tweets_prompt)}")
            return tweets_prompt

    def get_prompt(self, tweets: str):
        template = """Transform the provided tweets into a news article format suitable for publication in a newspaper. Each tweet should be restructured into a formal, concise, and informative paragraph. Maintain factual accuracy and neutrality, removing casual language, emojis, and hashtags. Enhance clarity by providing context where necessary, and ensure that the transitions between tweets are smooth, creating a coherent narrative. Aim for a professional tone that aligns with traditional journalism standards. Focus on the key information and relevant details to craft an engaging and informative piece. The more likes, the more important.\n \n{tweets}"""
        prompt = template.format(tweets=tweets)
        return prompt

    def get_total_tweets(self, query={}, **kwargs):
        if not query:
            response = self.client.collections["posts"].retrieve()
            num_documents = response["num_documents"]
        else:
            response = self.client.collections["posts"].documents.search(query)
            num_documents = response["found"]

        return num_documents

    def clean_tweet(self, tweet):
        tweet = clean(
            tweet,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=False,  # lowercase text
            no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en",  # set to 'de' for German special handling
        )
        return tweet


if __name__ == "__main__":
    twitter_prompt = TwitterPrompt()
    prompt = twitter_prompt()
    print(prompt)
