import torch
import json
from torch.utils.data import Dataset
import os
from PIL import Image
# from utils import pre_question
from dataset.utils import pre_question


class ImageDataset(Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, ann_file, transform, vqa_root, eos = '[SEP]', split = 'train', max_ques_words = 30, answer_list = ''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        
        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'val':
            self.max_ques_words = 50 # do not limit questions length during test
            self.answer_list = json.load(open(answer_list, 'r'))
            
    def __getitem__(self, index):
        
        ann = self.ann[index]
        
        # sample_path = COCO_val2014_000000581929.jpg -> 12 digits
        max_len = 12
        local_image_path = f"COCO_{self.split}2014_" + "0" * (max_len - len(str(ann['image_id']))) + str(ann['image_id']) + '.jpg'
        # set path
        image_path = os.path.join(self.vqa_root, f'{self.split}2014', local_image_path)
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            
            return image, question, question_id
        
        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)
            ans_weights = {}
            for answer in ann['anwers']:
                answer = answer['answer']
                if answer in ans_weights.keys():
                    ans_weights[answer] += 1/len(ann['anwers'])
                else:
                    ans_weights[answer] = 1/len(ann['anwers'])

            answers = list(ans_weights.keys())
            weights = list(ans_weights.values())

            answers = [answer + self.eos for answer in answers]

        return image, question, answers, weights

    def __len__(self):
        return len(self.ann)

# class ImageDataset():

# class 
# if __name__ == '__main__':
#     import yaml
#     args = 'configs/dataset.yaml'
#     config = yaml.load(open(args, 'r'), Loader=yaml.Loader)

#     train_dataset = ImageDataset(config['train_file'], transform = None, vqa_root=config['vqa_root'], split='train') 
#     print(train_dataset.__len__)
