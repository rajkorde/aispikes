# https://github.com/illuin-tech/colpali
import sys

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

model_name = "vidore/colqwen2-v1.0"


model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()


processor = ColQwen2Processor.from_pretrained(model_name)

images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]

queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

batch_images = processor.process_images(images, return_tensors="pt").to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
