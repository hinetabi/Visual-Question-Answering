import torch
import json
from torch.utils.data import Dataset
import os
from PIL import Image
# from utils import pre_question
from dataset.utils import pre_question


class ImageDatasetClassifier(Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, ann_file, transform, vqa_root, max_ques_words = 30, answer_list = ''):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        
        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.answer_list = json.load(open(answer_list, 'r'))

        
    def find_answer(self, answer_list, answer):
        for k, v in answer_list:
            if v == answer:
                return k
        return -1
    
    def __getitem__(self, index):
        
        ann = self.ann[index]
        max_len = 12
        local_image_path = f"COCO_{self.split}2014_" + "0" * (max_len - len(str(ann['image_id']))) + str(ann['image_id']) + '.jpg'
        # set path
        image_path = os.path.join(self.vqa_root, f'{self.split}2014', local_image_path)
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        question = pre_question(ann['question'], self.max_ques_words)

        answer = ann['multiple_choice_answer']
        answer = self.find_answer(answer=answer, answer_list=self.answer_list)

        return image, question, answer

    def __len__(self):
        return len(self.ann)
