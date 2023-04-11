import torch

if __name__ == '__main__':
    from dataset import create_dataset_classifier
    import yaml

    config = yaml.load(open('configs/dataset.yaml','r'), Loader=yaml.Loader)

    train, val = create_dataset_classifier(config)

    trainloader = torch.utils.DataLoader(train, batch_size=2, suffle=True)

    for i, (image, question, answer_num) in enumerate(trainloader):
        print(i)

        print(f'question : {question}, ans : {answer_num}')

        print(1)