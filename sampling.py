import json
from random import sample
from tqdm import tqdm

# return the json object as a dictionary after sampling the data, and saving the json object into the file
def sampling_data(url, saved_url, sampling_size):
    
    # get the validation data
    f = open(url)

    # returns JSON object as a dictionary
    data = json.load(f)
    print(len(data))

    # data train ['questions']
    image_ids = set()
    for question in data:
        image_ids.add(question['image_id'])
    
    # random
    image_ids = sample(list(image_ids), int(len(image_ids) * sampling_size))
    
    ans = []
    for question in data:
        if question['image_id'] in image_ids:
            ans.append(question)
    
    print(len(ans))
    
    with open(saved_url, "w") as outfile:
        json.dump(ans, outfile)
    # Closing file
    f.close()

def read_file_json(filename):
    f = open(filename, 'r')

    data = json.load(f)
    
    f.close()

    return data

# TODO: transfer the data -> data used for training
# annotations include image_id, question_id, answers -> train from this. 
# answer weights?
def format_annotations():
    for phase in ['train', 'val']:
        
        annotations = read_file_json(f'data/v2_mscoco_{phase}2014_annotations.json') # image id, question id, answer
        questions = read_file_json(f'data/v2_OpenEnded_mscoco_{phase}2014_questions.json') # image id, question id, question
        
        save_ann_for_training = []
        for annotation in tqdm(annotations['annotations']):
            for question in questions['questions']:
                if annotation['image_id'] == question['image_id']:
                    if annotation['question_id'] == question['question_id']:
                        save_ann_for_training.append({
                            "image_id" : annotation['image_id'],
                            "question_id" : annotation['question_id'],
                            'question_type': annotation['question_type'],
                            "question" : question['question'],
                            "answers" : annotation['answers'],
                            "data_subtype" : phase,
                            'multiple_choice_answer':annotation['multiple_choice_answer']
                        })

           
        with open(f'data/{phase}_formatted.json', 'w') as f:
            json.dump(save_ann_for_training, f)

def get_ans():
    ans = set()

    train = read_file_json("data/train_formatted.json")
    val = read_file_json("data/val_formatted.json")

    for i in tqdm(train):
        ans.add(i['multiple_choice_answer'])
    for i in tqdm(val):
        ans.add(i['multiple_choice_answer'])
    dic  =  {}
    for i, an in enumerate(ans):
        dic [i] = an

    with open(f'data/vqa_dict.json', 'w') as f:
            json.dump(dic, f)

def sampling_by_ans():
    ans = read_file_json("data/vqa_dict.json")
    ans = ans.values()
    trains = read_file_json("data/train_formatted.json")
    vals = read_file_json("data/val_formatted.json")

    sample_train = []
    for train in tqdm(trains):
        if train['multiple_choice_answer'] in ans:
            sample_train.append(train)


    sample_val = []
    for val in tqdm(vals):
        if val['multiple_choice_answer'] in ans:
            sample_val.append(val)
            
    dic = {}
    dic["sample_train"] = sample_train
    dic["sample_val"] = sample_val
    
    for phase in ["sample_train", "sample_val"]:
        with open(f'data/{phase}_formatted.json', 'w') as f:
            json.dump(dic[phase], f)

if __name__ == '__main__':
    # for phase in ['train', 'val']:
    #     sampling_data(url=f'data/{phase}_formatted.json', saved_url=f'data/sample_{phase}_formatted.json', sampling_size=0.1)    
    # format_annotations()

    # get_ans()
    sampling_by_ans()








