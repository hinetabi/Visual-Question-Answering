from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from model.model_vqa import ALBEF
import yaml
from transformers import BertTokenizer
from utils import init_tokenizer


def predict(model, image_path, question, device, config):
    im = load_image(image_path, image_size=384, device=device, config=config)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        answer = model(im, question, generate = True, train = False)
        return 'Answer: ' + answer[0]

def load_image(image, image_size, device, config):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ]) 

    # transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # ])

    # image = transform(raw_image).unsqueeze(0).to(device)
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

if __name__ == '__main__':
    config_path = './configs/dataset.yaml'
    text_encoder = 'bert-base-uncased'
    text_decoder = 'bert-base-uncased'
    tokenizer = init_tokenizer(text_encoder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    print("Creating model")
    model = ALBEF(config=config, text_encoder=text_encoder, text_decoder=text_decoder, tokenizer=tokenizer)
    model = model.to(device)  

    ans = predict(model, image_path='data/val2014/COCO_val2014_000000000042.jpg', question='What is this?', device=device, config = config)
    print(ans)

