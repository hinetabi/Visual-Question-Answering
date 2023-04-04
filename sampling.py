 
import json
  
# return the json object as a dictionary after sampling the data, and saving the json object into the file
def sampling_data(url, sampling_size):
    
    # get the validation data
    f = open(url)

    # returns JSON object as a dictionary
    data = json.load(f)

    # data train ['questions']
    image_ids = set()
    for question in data['questions']:
        image_ids.add(question['image_id'])
    
    # random
    # image_ids = list(image_ids)
    ans = {}
    for question in data['questions']:
        if question['image_id'] in image_ids:
            ans = 
            
    # Closing file
    f.close()


















