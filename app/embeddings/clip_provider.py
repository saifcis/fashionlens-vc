
import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, CLIPModel


class CLIPProvider:

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):

        self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME,
            local_files_only=True
        )

        self.model = CLIPModel.from_pretrained(
            self.MODEL_NAME,
            local_files_only=True
        ).to(self.device)

        self.model.eval()

    def _normalize(self, features):

        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        if features.shape[-1] != 512:
            raise ValueError("Unexpected embedding dimensions")

        return F.normalize(
            features.float(),
            p=2,
            dim=-1
        ).cpu().numpy()

    def encode_image(self, image_path):

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        return self.encode_images([image])[0]

    def encode_images(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)

            return self._normalize(features)

    def encode_text(self, text):

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)

            return self._normalize(features)[0]

