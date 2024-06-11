import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Moderation:
    def __init__(self):
        self.moderation_model = AutoModelForSequenceClassification.from_pretrained(
            "KoalaAI/Text-Moderation"
        )
        self.moderation_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")

    @torch.inference_mode()
    def __call__(self, prompt, threshold=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.moderation_model(**inputs)

        # Get the predicted logits
        logits = outputs.logits

        # Apply softmax to get probabilities (scores)
        probabilities = logits.softmax(dim=-1).squeeze()
        probabilities = logits.softmax(dim=-1).squeeze()

        # Retrieve the labels
        id2label = self.moderation_model.config.id2label
        labels = [id2label[idx] for idx in range(len(probabilities))]

        # Combine labels and probabilities, then sort
        label_prob_pairs = list(zip(labels, probabilities))
        label_prob_pairs.sort(key=lambda item: item[1], reverse=True)
        max_label = label_prob_pairs[0][0]
        if max_label == "OK":
            is_flagged = False
        else:
            is_flagged = True
        return is_flagged, label_prob_pairs
