import torch
import json
from torch.utils.data import Dataset
import os
from PIL import Image
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

        if split == 'test':
            self.max_ques_words = 50 # do not limit questions length during test
            self.answer_list = json.load(open(answer_list, 'r'))
            
    def __getitem__(self, index):
        
        ann = self.ann[index]
        
        # vqa datasets
        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.vqa_root, ann['image'])
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            
            return image, question, question_id
        
        elif self.split == 'train':
            question = pre_question(ann['question', self.max_ques_words])
            ans_weights = {}
            for answer in ann['answers']:
                if answer in ans_weights.key():
                    ans_weights[answer] += 1/len(ann['answer'])
                else:
                    ans_weights[answer] = 1/len(ann['answer'])

            answers = list(ans_weights.keys())
            weights = list(ans_weights.values())

            answers = [answer + self.eos for answer in answers]

        return image, question, answers, weights

    def __len__(self):
        return len(self.ann)

# class ImageDataset():

# class 
    