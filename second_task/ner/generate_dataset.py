import json
import random
import os

# List of animals that the NER model should recognize and tag
ANIMALS = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]

# List of fictional/unknown animals that should NOT be recognized as valid animals
# These are used to test model's ability to identify out-of-distribution animal names
UNKNOWN_ANIMALS = [
    "dragon",
    "unicorn",
    "phoenix",
    "griffin",
    "dinosaur"
]

# Sentence templates for data generation
# These templates are used to create diverse training examples

# Templates with one animal 
# Used to generate sentences containing a single animal
SINGLE_TEMPLATES = [

"There is a {animal} in the picture.",
"I see a {animal} in this photo.",
"The image clearly shows a {animal}.",
"Look at this cute {animal}.",
"The {animal} is standing in the field.",
"What a beautiful {animal}.",
"A wild {animal} appeared in the forest.",
"The camera captured a {animal}.",
"Someone photographed a {animal}.",
"A {animal} is walking near the river.",

"The {animal} looks very calm.",
"I think that is a {animal}.",
"A small {animal} can be seen here.",
"There seems to be a {animal} in the background.",
"The {animal} is looking directly at the camera.",
"A large {animal} dominates the scene.",
"A curious {animal} is exploring the area.",
"This picture contains a {animal}.",
"Do you notice the {animal} in this image?",
"The photo probably contains a {animal}.",

"Many people believe this is a {animal}.",
"The photographer managed to capture a {animal}.",
"A {animal} appears in the center of the image.",
"Right in the middle there is a {animal}.",
"The picture might show a {animal}.",
"At first glance it looks like a {animal}.",
"In the corner of the image there is a {animal}.",
"Somewhere in the scene a {animal} is visible.",
"I believe the animal here is a {animal}.",
"Could this be a {animal}?",

"Perhaps the image contains a {animal}.",
"It looks like a {animal} to me.",
"That might actually be a {animal}.",
"The object in the photo resembles a {animal}.",
"A {animal} can be spotted easily.",
"The scene features a {animal}.",
"The main subject is a {animal}.",
"This image likely includes a {animal}.",
]

# Templates with two animals 
# Used to generate sentences containing multiple animals for more complex examples
MULTI_TEMPLATES = [

"I see a {animal1} and a {animal2} in the picture.",
"There is a {animal1} next to a {animal2}.",
"A {animal1} is standing near a {animal2}.",
"The photo shows both a {animal1} and a {animal2}.",
"Two animals appear: a {animal1} and a {animal2}.",
"A {animal1} is chasing a {animal2}.",
"The image contains a {animal1} together with a {animal2}.",
"A {animal1} and a {animal2} are visible.",
"Both a {animal1} and a {animal2} can be seen here.",
"The scene includes a {animal1} along with a {animal2}.",

]

# Templates with no animals
# Used to generate negative examples so the model learns when NOT to tag
NO_ANIMAL_TEMPLATES = [

"There is nothing interesting in the picture.",
"I cannot see any animal here.",
"This photo shows only nature.",
"There are no animals in this image.",
"The scene is empty.",
"Only grass and trees are visible.",
"This picture contains no animals.",
"The image shows a landscape.",
"I only see the sky and clouds.",
"No animals appear in this photo."
]

# Common objects that are NOT animals
# Used in templates to create sentences where the model should not tag words as animals
OBJECTS = [
    "tree",
    "rock",
    "car",
    "house",
    "flower",
    "river",
    "cloud",
    "grass",
    "building",
    "road"
]


def tokenize(text):
    """
    Split text into tokens by removing punctuation and splitting on whitespace.
    """
    return text.replace(".", "").replace("?", "").split()


def create_labels(tokens, animals_in_sentence):
    """
    Create BIO (Begin-Inside-Outside) labels for NER task.
    """
    # Initialize all tokens as 'O' (Outside - not an entity)
    labels = ["O"] * len(tokens)

    # Tag tokens that match animals in the sentence
    for animal in animals_in_sentence:
        for i, token in enumerate(tokens):
            if token.lower() == animal:
                # Mark this token as beginning of an animal entity
                labels[i] = "B-ANIMAL"

    return labels


def generate_sample():
    """
    Generate a single training sample with tokens and labels.
    """
    # Randomly determine what type of sample to generate
    choice = random.random()

    # Single known animal
    if choice < 0.6:
        template = random.choice(SINGLE_TEMPLATES)
        animal = random.choice(ANIMALS)
        sentence = template.format(animal=animal)
        animals_in_sentence = [animal]

    # Two animals 
    elif choice < 0.85:
        # Select random template and two different animals
        template = random.choice(MULTI_TEMPLATES)
        animal1, animal2 = random.sample(ANIMALS, 2)
        sentence = template.format(
            animal1=animal1,
            animal2=animal2
        )
        animals_in_sentence = [animal1, animal2]

    # Unknown/fictional animal (should not be tagged)
    elif choice < 0.95:
        template = random.choice(SINGLE_TEMPLATES)
        animal = random.choice(UNKNOWN_ANIMALS)
        sentence = template.format(animal=animal)
        animals_in_sentence = []  

    # Random object (not an animal)
    elif choice < 0.98:
        template = random.choice(SINGLE_TEMPLATES)
        obj = random.choice(OBJECTS)
        sentence = template.format(animal=obj)
        animals_in_sentence = []  

    # No animals at all
    else:
        template = random.choice(NO_ANIMAL_TEMPLATES)
        sentence = template
        animals_in_sentence = []

    # Tokenize the sentence and create labels
    tokens = tokenize(sentence)
    labels = create_labels(tokens, animals_in_sentence)

    return {
        "tokens": tokens,
        "labels": labels
    }


def generate_dataset(size):
    """
    Generate a dataset of specified size.
    """
    return [generate_sample() for _ in range(size)]


def save_dataset(train_size=8000, val_size=1000):
    """
    Generate and save both training and validation datasets.
    """
    # Generate datasets
    train_data = generate_dataset(train_size)
    val_data = generate_dataset(val_size)

    # Create output directory
    os.makedirs("data/ner", exist_ok=True)

    # Save training data
    with open("data/ner/train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    # Save validation data
    with open("data/ner/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    # Print summary
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print("Dataset saved to data/ner/")


if __name__ == "__main__":
    # Generate and save the default dataset
    save_dataset()