 
import json
from random import sample
from tqdm import tqdm

# return the json object as a dictionary after sampling the data, and saving the json object into the file
def sampling_data(url, saved_url, sampling_size):
    
    # get the validation data
    f = open(url)

    # returns JSON object as a dictionary
    data = json.load(f)
    print(len(data['questions']))

    # data train ['questions']
    image_ids = set()
    for question in data['questions']:
        image_ids.add(question['image_id'])
    
    # random
    image_ids = sample(list(image_ids), int(len(image_ids) * sampling_size))
    
    ans = {}
    ans['questions'] = []
    temp = {}
    for question in data['questions']:
        if question['image_id'] in image_ids:
            ans['questions'].append({
                'image_id' : question['image_id'],
                'question' : question['question'],
                'question_id' : question['question_id']
            })
    
    print(len(ans['questions']))
    
    with open(saved_url, "w") as outfile:
        json.dump(ans, outfile)
    # Closing file
    f.close()

def read_file_json(filename):
    f = open(filename, 'r')

    data = json.load(f)
    
    f.close()

    return data
    

# read_file_json("data/v2_mscoco_train2014_annotations.json") 
# read_file_json("data/v2_OpenEnded_mscoco_train2014_questions.json") 

# sampling_data('data\\v2_Questions_Val_mscoco\\v2_OpenEnded_mscoco_val2014_questions.json', 'data/sample_val.json', 0.1)
# sampling_data('data\\v2_Questions_Train_mscoco\\v2_OpenEnded_mscoco_train2014_questions.json', 'data/sample_train.json', 0.1)


# TODO: transfer the data -> data used for training
# annotations include image_id, question_id, answers -> train from this. 
# answer weights?
if __name__ == '__main__':
    for ln in ['train', 'val']:
        
        annotations = read_file_json(f'data/v2_mscoco_{ln}2014_annotations.json') # image id, question id, answer
        questions = read_file_json(f'data/v2_OpenEnded_mscoco_{ln}2014_questions.json') # image id, question id, question
        
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
                            "anwers" : annotation['answers'],
                            "data_subtype" : ln
                        })

        
        with open(f'data/{ln}_formatted.json', 'w') as f:
            json.dump(save_ann_for_training, f)

        print(f'data/{ln}_formatted is saved!')













