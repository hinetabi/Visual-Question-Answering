import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import os
import numpy as np
import random
from pathlib import Path
from model.model_vqa import ALBEF
from transformers import BertTokenizer
from dataset import create_dataset, create_loader, vqa_collate_fn
from torch.optim import AdamW
import time
import datetime
from tqdm import tqdm
import wandb
import json

import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, tokenizer, epoch, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    # loop over the dataset multiple times
    running_loss = 0.0
    for i,(image, question, answer, weights, n) in tqdm(enumerate(metric_logger.log_every(train_loader, print_freq, header))):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        loss = model(image, question_input, answer_input, train=True, alpha=config['alpha'], k=n, weights=weights)        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss.backward()
        optimizer.step()

        # log to terminal
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    
    running_loss = running_loss / i
    print("Averaged stats:", metric_logger.global_avg())         
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def test(model, train_loader, optimizer, tokenizer, epoch, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    # loop over the dataset multiple times
    running_loss = 0.0
    for i,(image, question, answer, weights, n) in tqdm(enumerate(metric_logger.log_every(train_loader, print_freq, header))):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        with torch.no_grad():
            loss = model(image, question_input, answer_input, train=False, alpha=config['alpha'], k=n, weights=weights)        
        # log to terminal
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    
    running_loss = running_loss / i
    print("Averaged stats:", metric_logger.global_avg())         
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 



@torch.no_grad()
def evaluation(model, test_loader, tokenizer, config):
    # test

    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter= " ")
    header = "Generate VQA test results"
    print_freq = 50
    
    result = []
    
    # answer list not exit!!!!!!!!
    answer_list = [answer+config['eos'] for answer in test_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":test_loader.dataset.answer_list[topk_id[pred]]})   

    return result
    

def main(args, config):
    # config
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create dataset
    datasets = create_dataset(config)
    # create dataloader
    train_loader, test_loader = create_loader(datasets, samplers=[None, None], 
                                            batch_size=[config['batch_size_train'], config['batch_size_test']], 
                                            num_workers=[4,4], 
                                            is_trains=[True, False],
                                            collate_fns=[vqa_collate_fn, None])
    
    # tokenizer for questions and answers
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)
    
    # epoch
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    # warmup_epoch = config['schedular']['warmup_epochs']
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr = float(config['optimizer']['lr']), weight_decay = float(config['optimizer']['weight_decay']))
    # training
    print("Start training")
    start_time = time.time()

    # wandb init
    run = wandb.init(
    # Set the project where this run will be logged
    project="visual question answering",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": config['optimizer']['lr'],
        "epochs": config['schedular']['epochs'],
    })

    model_without_ddp = model
    
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            a = 0

        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        if not args.evaluate:
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, config)
            test_stats = train(model, test_loader, optimizer, tokenizer, epoch, config)
            # log to wandb
            wandb.log({
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
            })
            
            wandb.log({**{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                        })
            
        
        if args.evaluate:
            break
            
        # save log for training
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
        with open(os.path.join(args.output_dir, "train", "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")                        
                        
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, "train", 'checkpoint_%02d.pth'%epoch))
        
        # save log for testing
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                        }                
        with open(os.path.join(args.output_dir,"test", "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")                        
                        
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, "test", 'checkpoint_%02d.pth'%epoch))

    # evaluating
    # vqa_result = eval(model, test_loader, tokenizer, config)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('Training time {}'.format(total_time_str)) 
    torch.save(model, os.path.join(config['output_dir'], "model.pth"))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/dataset.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/albef')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # update arguments into config
    for k,v in args._get_kwargs():
        config[k] = v    

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)

    # test loader need answer argument for running
    # get to know about how model bert decode, run debug?