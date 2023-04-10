import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
import os
from dataset import create_dataset_classifier, create_loader
from transformers import BertTokenizer
from model.clip import QuestionAnswerClassifier
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(epoch, model, trainloader, optimizer,  criterion, tokenizer):
    # loop over the dataset multiple times
    running_loss = 0.0
    print(f'Training epoch {epoch}: ')

    model.train()

    for  i,(image, question, labels) in tqdm(enumerate(trainloader, 0)):
        
        image, question = image.to(device), question.to(device)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        # zero the parameter gradients
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # forward + backward + optimize
            outputs = model(image, question_input)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            _, preds = torch.max(outputs, 1)
        
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        

    print('Loss: {}'.format(running_loss))
    print('Acc: {}%'.format(running_corrects))

    print('Finished Training')
    
    return running_loss
    
def main(config):
    # dataloader = Dataloader(config
    datasets = create_dataset_classifier(config=config)
    
    train_loader, val_loader = create_loader(datasets, samplers=[None, None], 
                                            batch_size=[config['batch_size_train'], config['batch_size_test']], 
                                            num_workers=[4,4], 
                                            is_trains=[True, False],
                                            collate_fns=[None, None])
    
    # tokenizer for questions and answers
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    
    
    #### Model #### 
    print("Creating model")
    model = QuestionAnswerClassifier(config=config, text_encoder=args.text_encoder)
    model = model.to(device)
    
    # loss
    loss = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(params = model.parameters(), lr = 1e-5)
    
    for i in range(config['epoch']):
        train_one_epoch(trainloader=train_loader, criterion = loss, optimizer= optimizer, tokenizer=tokenizer)

    return 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/dataset.yaml') 
    parser.add_argument('--output_dir', default='output/albef')
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
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)