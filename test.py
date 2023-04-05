class test:
    def __init__(self):
        self.split = 'train'
        ann = {"image_id" : 581929}
        max_len = 12
        self.local_image_path = f"COCO_{self.split}2014_" + "0" * (12 - len(str(ann['image_id']))) + str(ann['image_id']) + '.jpg'
    
    def get(self):
        return self.local_image_path

test = test()
print(test.get())