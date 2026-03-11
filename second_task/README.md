# Task 2: Multimodal Animal Matching Pipeline

## Overview

This project implements a **multimodal machine learning pipeline** that combines information from **text and images** to determine whether they refer to the same animal.

The pipeline consists of two main components:

1. **Named Entity Recognition (NER)** model that extracts animal names from text.
2. **Image classification model** that predicts the animal shown in an image.

The system compares the animal mentioned in the text with the animal predicted from the image and returns whether they match.

Example input:

```
python pipeline.py --text "It is cat" --image "images/cat.jpg"
```
Example output:
```
Step 1: NER
Animals extracted from text: cat

Step 2: Image Classification
Predicted class: cat
Confidence: 1.000

Step 3: Matching
Final Result: True
```

# Project Structure
second_task/    
├── data    
│   ├── animals10 # image dataset (not included in repository)  
│   ├── ner # generated NER dataset 
│   │   ├── train.json  
│   │   └── val.json    
│   └── prepare_dataset.py # generated NER dataset  
├── eda 
│   └── animal_dataset_eda.ipynb    
├── images  
├── models  
│   ├── ner_model # trained NER model (Git LFS) 
│   │   ├── config.json 
│   │   ├── model.safetensors   
│   │   ├── tokenizer_config.json   
│   │   ├── tokenizer.json  
│   │   └── training_args.bin   
│   └── vision_model.pth # trained vision model (Git LFS)   
├── ner 
│   ├── generate_dataset.py # synthetic dataset generation  
│   ├── inference.py    
│   └── train.py    
├── vision  
│   ├── inference.py    
│   └── train.py    
├── .gitattributes  
├── .gitignore  
├── demo.ipynb # pipeline demonstration 
├── pipeline.py # multimodal pipeline   
├── README.md   
└── requirements.txt    


---

# Setup Instruction

1. Clone the repository
```
git clone https://github.com/OAtamanchuk/winstars-ds-test.git
cd winstars-ds-test/first_task
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Install Git LFS

Pretrained models in this repository are stored using **Git Large File Storage (Git LFS)**.

Install Git LFS:
```
git lfs install
```

After cloning the repository, download the model weights:
```
git lfs pull
```

This will download:
- models/ner_model/model.safetensors
- models/vision_model.pth

# Running the Pipeline

Once the models are downloaded, you can run the pipeline directly.

Example:
```
python pipeline.py --text "I see a cat" --image images/cat.jpg
```

Example output:
```
Animals extracted from text: ['cat']

Predicted animal from image: cat
Prediction confidence: 1.0

Final result: True
```

# Demo Notebook

The repository also contains a demonstration notebook **demo.ipynb**.

The notebook demonstrates:

- correct matches between text and image
- mismatched examples
- cases with multiple animals
- cases where no animal is detected in text

It can be used to interactively test the pipeline.

# Training the Models

If you want to **retrain the models from scratch**, follow the steps below.

# Datasets

## Image Dataset

The image classifier is trained on the **Animals10 dataset**.

Download it from Kaggle:

https://www.kaggle.com/datasets/alessiocorrado99/animals10

Place the dataset into **data/animals10/raw-img**

Expected structure: 
data/
├── animals10
│   └── raw-img
│       ├── butterfly
│       ├── cat
│       ├── chicken
│       ├── cow
│       ├── dog
│       ├── elephant
│       ├── horse
│       ├── sheep
│       ├── spider
│       └── squirrel
└── prepare_dataset.py

## Dataset Preparation

Run the dataset preparation script:
```
python data/prepare_dataset.py
```

This script prepares the directory structure required for training the vision model.

## NER Dataset

The NER dataset is **synthetically generated** using templates that contain animal names.

Generate the dataset with:
```
python ner/generate_dataset.py
```
This will create:
- data/ner/train.json
- data/ner/val.json

# Model Training

## Train the Vision Model

The image classification model can be trained using:
```
python vision/train.py 
```
By default, the script uses the dataset located at:
```
data/animals10/raw-img
```
Available parameters:

| Parameter      | Description                                       | Default                  |
| -------------- | ------------------------------------------------- | ------------------------ |
| `--data_dir`   | Directory containing image folders for each class | `data/animals10/raw-img` |
| `--epochs`     | Maximum number of training epochs                 | 15                       |
| `--batch_size` | Batch size for training and validation            | 32                       |
| `--lr`         | Learning rate for the optimizer                   | 0.0005                   |
| `--patience`   | Early stopping patience                           | 4                        |


Example training command:
```
python vision/train.py \
--data_dir data/animals10/raw-img \
--epochs 20 \
--batch_size 32 \
--lr 0.001 \
--patience 5
```

The trained model will be saved to **models/vision_model.pth**

## Train the NER Model

The NER model can be trained using:
```
python ner/train.py
```
Available parameters:
| Parameter      | Description                        | Default               |
| -------------- | ---------------------------------- | --------------------- |
| `--train_path` | Path to training dataset           | `data/ner/train.json` |
| `--val_path`   | Path to validation dataset         | `data/ner/val.json`   |
| `--epochs`     | Number of training epochs          | 4                     |
| `--batch_size` | Training and evaluation batch size | 8                     |
| `--lr`         | Learning rate                      | 2e-5                  |

Example training command:
```
python ner/train.py \
--train_path data/ner/train.json \
--val_path data/ner/val.json \
--epochs 5 \
--batch_size 16 \
--lr 2e-5
```

The trained model will be saved to **models/ner_model/**

# Models Used

The multimodal pipeline combines two machine learning models: one for **image classification** and one for **named entity recognition**.

## Vision Model

The image classification component is based on a pretrained convolutional neural network.

Architecture:   
**ResNet18** from
PyTorch and Torchvision.

The model is initialized with ImageNet pretrained weights and fine-tuned on the Animals10 dataset.

Input:

- 224x224 RGB image

Output:

- animal class
- confidence score

## NER Model

The NER component is based on the transformer model **DistilBERT**.

The model is fine-tuned for token classification to identify animal names in text.

Input:

- "I saw a dog and a cat"

Output:

- ["dog", "cat"]

## Multimodal Decision Step

The final pipeline compares animal_from_text == animal_from_image.  
if both animals match - the pipeline returns **True**, otherwise - **False**.




