from src.predict import classifier_service
from src.config import MODEL_PATH, CLASS_NAMES

def interactive_demo():
    """
    Loads the model once, enters an interactive loop where the
    user can type or paste news text and receive predictions.

    Usage:
        python run_demo.py
    """
    print("\n" + "=" * 55)
    print("  CURB: Classification Using Refined BERT")
    print("  News Article Classifier (CLI Demo)")
    print("=" * 55)

    print("Loading model...")
    success = classifier_service.load_model(MODEL_PATH)

    if not success:
        print(f"Failed to load model: {classifier_service.model_load_error}")
        print("Make sure you have trained the model first: python -m src.train")
        return

    print(f"Model loaded in {classifier_service.model_load_time:.2f}s")
    print(f"Categories: {', '.join(CLASS_NAMES)}")
    print("\nEnter news text to classify (or 'quit' to exit).")
    print("-" * 55)

    while True:
        try:
            text = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text: continue

        if text.lower() in ("quit", "exit", "q"): break

        try:
            result = classifier_service.predict(text)

            print(f"\n  Category:   {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.1%}")

            print("\n  All probabilities:")
            for category, prob in result["sorted_predictions"]:
                if prob >= 0.034:
                    bar = "█" * int(prob * 30)
                elif prob >= 0.017:
                    bar = "▌"
                elif prob >= 0.0084:
                    bar = "▎"
                else:
                    bar = "▏"

                print(f"    {category:<10} {prob:.3f}  {bar}")

            print()
        except ValueError as e:
            print(f"\n  Validation error: {e}")
        except Exception as e:
            print(f"\n  Prediction failed: {e}")

    print("Goodbye!")

if __name__ == '__main__':
    interactive_demo()