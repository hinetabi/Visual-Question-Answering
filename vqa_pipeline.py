

from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BertTokenizerFast
import requests
from PIL import Image

# prepare image + question
url = 'data/train2014/COCO_train2014_000000000009.jpg'
image = Image.open(url)
text = "How many rice components in the disk?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
print("Predicted answer: ", processor.tokenizer.decode(idx))

# eval



