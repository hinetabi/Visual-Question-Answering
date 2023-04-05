from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset.randaugment import RandomAugment
from torchvision import transforms

def create_dataset(dataset, config):
    normalize = transforms.Normalize()

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, batch_size, num_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last =  True
        else:
            shuffle = False
            drop_last = False
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