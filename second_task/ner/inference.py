import torch
from transformers import pipeline


def load_ner_model(model_dir="models/ner_model"):
    """
    Load trained NER pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1

    # Create NER pipeline with the trained model
    # aggregation_strategy="simple" merges subword tokens into complete words
    ner_pipeline = pipeline(
        "ner",
        model=model_dir,
        tokenizer=model_dir,
        device=device,
        aggregation_strategy="simple"
    )

    return ner_pipeline


def extract_animals(text, ner_pipeline):
    """
    Extract ANIMAL entities from text.
    """
    results = ner_pipeline(text)

    animals = []

    # Process each detected entity
    for entity in results:
        # Check if the entity is tagged as an ANIMAL
        if entity["entity_group"] == "ANIMAL":
            # Clean up the word by removing subword token markers
            clean_word = entity["word"].replace(" ##", "").replace("##", "").strip()
            animals.append(clean_word)

    # Return the list of extracted animal names
    return animals