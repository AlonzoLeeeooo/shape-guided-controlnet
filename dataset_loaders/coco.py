import os
import random
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.image_path = args.image_path
        self.caption_path = args.caption_path
        self.condition_path = args.condition_path            
        self.image_file_list = os.listdir(args.image_path)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )
        self.use_null_text = args.use_null_text
        self.tokenizer = tokenizer

        # Load COCO annotation file
        with open(self.caption_path, 'r') as f:
            annotations = json.load(f)['annotations']
            
        self.imageid_caption_dict = {}
        for annotation in annotations:
            complete_imageid = str(annotation['image_id']).rjust(12, '0')
            self.imageid_caption_dict[complete_imageid] = annotation['caption']
            
        print(f"There are {len(self.imageid_caption_dict.keys())} annotated captions in total.")
                
        
        
    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

        
    def __len__(self):
        return len(self.image_file_list)
    
    
    def __getitem__(self, index):
        image_file_path = self.image_file_list[index]
        image = Image.open(os.path.join(self.image_path, image_file_path)).convert('RGB')
        filename = os.path.basename(image_file_path).split('.')[0]
        conditioning_image = Image.open(os.path.join(self.condition_path, filename + '.png')).convert('RGB')
        image = self.image_transforms(image)
        conditioning_image = self.conditioning_image_transforms(conditioning_image)
        
        # Obtain `image_id`
        imageid = filename.replace('COCO_train2014_', '')
        
        if self.use_null_text:
            text_captions = [""]
        else:
            text_captions = []
            annotated_caption = self.imageid_caption_dict[imageid]
            text_captions.append(annotated_caption)
            

        input_ids = self.tokenize_captions(text_captions)

        batch = {}
        batch['pixel_values'] = image
        batch['conditioning_pixel_values'] = conditioning_image
        batch['input_ids'] = input_ids

        return batch