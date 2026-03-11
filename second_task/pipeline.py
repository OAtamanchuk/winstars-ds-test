import argparse

from ner.inference import load_ner_model, extract_animals
from vision.inference import load_vision_model, predict_image


def pipeline(text, image_path, ner_model, vision_model, classes):
    """
    Main pipeline function.
    """

    # Step 1: perform named-entity recognition on the text
    print("\nStep 1: NER")

    animals = extract_animals(text, ner_model)

    if animals:
        print("Animals extracted from text:", ", ".join(animals))
    else:
        # nothing to compare against, but still proceed to classification
        print("No animals detected in the text.")

    # Step 2: classify the provided image with the vision model
    print("\nStep 2: Image Classification")

    label, confidence = predict_image(image_path, vision_model, classes)

    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.3f}")

    # Step 3: compare prediction against text animals
    print("\nStep 3: Matching")

    # Normalize all names to lowercase for case-insensitive matching
    animals_lower = [a.lower() for a in animals]

    # Matching results - True if label exists among extracted names
    match = label.lower() in animals_lower

    return match


def main():
    """
    Command-line entry point.
    Parses arguments, loads models, executes the pipeline, and prints the result.
    """

    parser = argparse.ArgumentParser(description="Animal recognition pipeline")

    parser.add_argument(
        "--text",
        required=True,
        help="Input text containing animal names"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file"
    )

    args = parser.parse_args()

    # Load models once before running pipeline
    print("Loading models.")

    ner_model = load_ner_model()
    vision_model, classes = load_vision_model()

    result = pipeline(
        text=args.text,
        image_path=args.image,
        ner_model=ner_model,
        vision_model=vision_model,
        classes=classes
    )

    print("Final Result:", result)


if __name__ == "__main__":
    main()