from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset.randaugment import RandomAugment
from torchvision import transforms
from dataset.dataset import ImageDataset
from dataset.dataset_classifier import ImageDatasetClassifier

def create_dataset_classifier(config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    # def __init__(self, ann_file, transform, vqa_root, eos = '[SEP]', split = 'train', max_ques_words = 30, answer_list = ''):


    train_dataset = ImageDatasetClassifier(config['train_file'], train_transform, config['vqa_root'], answer_list=config['vqa_dict']) 
    vqa_test_dataset = ImageDatasetClassifier(config['test_file'], test_transform, config['vqa_root'], answer_list=config['vqa_dict'])       
    return train_dataset, vqa_test_dataset

def create_dataset(config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    # def __init__(self, ann_file, transform, vqa_root, eos = '[SEP]', split = 'train', max_ques_words = 30, answer_list = ''):


    train_dataset = ImageDataset(config['train_file'], train_transform, config['vqa_root'], split='train') 
    vqa_test_dataset = ImageDataset(config['test_file'], test_transform, config['vqa_root'], split='val', answer_list=config['answer_list'])       
    return train_dataset, vqa_test_dataset
    
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, batch_size, num_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last =  True
        else:
            shuffle = False
            drop_last = True
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
        loaders.append(loader)
    return loaders


# transform a batch for load data into gpu
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    
    # torch.stack -> convert a list of n matrix a * b -> a matrix n * a * b
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n