import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
import os
from dataset import create_dataset_classifier, create_loader, vqa_collate_fn_classifier
from transformers import BertTokenizer
from model.vlit import QuestionAnswerClassifier
from tqdm import tqdm
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(epoch, model, trainloader, optimizer,  criterion, tokenizer):
    # loop over the dataset multiple times
    running_loss = 0.0
    print(f'Training epoch {epoch}: ')

    model.train()
    running_corrects = 0

    for  i,(image, question, labels) in tqdm(enumerate(trainloader)):
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        
        image, question_input = image.to(device), question_input.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # forward + backward + optimize
            outputs = model(image, question_input)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_corrects += (torch.sum(preds == labels) / config['batch_size_test'])

            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        print('Loss: {}'.format(loss.item()))
        wandb.log({
                f'Train_iter loss': loss,
                'iter': i,
            })
        
    acc = running_corrects / len(trainloader)
    loss = running_loss / len(trainloader)
    print('Train Acc: {}'.format(acc))
    print('Train Epoch Loss: {}'.format(loss))
    wandb.log({ f'train loss epoch {epoch}': loss,
                f'train acc epoch {epoch}' : acc,
                'epoch': epoch })

    # print('Acc: {}%'.format(running_corrects))
    save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
    torch.save(save_obj, os.path.join(args.output_dir, "train", 'checkpoint_%02d.pth'%epoch))

    print('Finished Training')
    
    return running_loss


def eval_one_epoch(epoch, model, testloader, optimizer,  criterion, tokenizer):
    # loop over the dataset multiple times
    running_loss = 0.0
    print(f'Evaluate epoch {epoch}: ')

    model.train()
    running_corrects = 0

    for  i,(image, question, labels) in tqdm(enumerate(testloader)):
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        
        image, question_input = image.to(device), question_input.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            # forward + backward + optimize
            outputs = model(image, question_input)
            
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_corrects += (torch.sum(preds == labels) / config['batch_size_test'])

        running_loss += loss.item()
        wandb.log({
                f'Val_iter loss': loss,
                'iter': i,
            })
        
    acc = running_corrects / len(testloader)
    loss = running_loss / len(testloader)
    print('Val Acc: {}'.format(acc))
    print('Val Epoch Loss: {}'.format(loss))
    wandb.log({ f'val loss epoch {epoch}': loss,
                f'val acc epoch {epoch}' : acc,
                'epoch': epoch })

    # print('Acc: {}%'.format(running_corrects))
    save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
    torch.save(save_obj, os.path.join(args.output_dir, "test", 'checkpoint_%02d.pth'%epoch))

    print('Finished Training')
    return running_loss

    
def main(args, config):

    # wandb init
    run = wandb.init(
    # Set the project where this run will be logged
    project="visual question answering",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": config['optimizer']['lr'],
        "epochs": config['schedular']['epochs'],
        
    },
    name= "vqa_vlit_classifier")


    # dataloader = Dataloader(config
    datasets = create_dataset_classifier(config=config)
    
    train_loader, val_loader = create_loader(datasets, samplers=[None, None], 
                                            batch_size=[config['batch_size_train'], config['batch_size_test']], 
                                            num_workers=[4,4], 
                                            is_trains=[True, False],
                                            collate_fns=[vqa_collate_fn_classifier, vqa_collate_fn_classifier])
    
    # tokenizer for questions and answers
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    
    
    #### Model #### 
    print("Creating model")
    model = QuestionAnswerClassifier(config=config, text_encoder=args.text_encoder)
    model = model.to(device)
    
    # loss
    loss = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(params = model.parameters(), lr = float(config['optimizer']['lr']))
    
    for epoch in range(int(config['epoch'])):
        train_stats = train_one_epoch(model = model, epoch = epoch, trainloader=train_loader, criterion = loss, optimizer= optimizer, tokenizer=tokenizer)
        torch.save(model.state_dict(), "output/vilt_classifier.pth")
        test_stats = eval_one_epoch(model=model, epoch=epoch, testloader=val_loader, criterion=loss, optimizer=optimizer, tokenizer=tokenizer)

    wandb.finish()
    torch.save(model.state_dict(), "output/vilt_classifier.pth")
    return 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/dataset.yaml') 
    parser.add_argument('--output_dir', default='output/vilt_vqa')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--epoch', default='5')
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # update arguments into config
    for k,v in args._get_kwargs():
        config[k] = v    

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    main(args, config)