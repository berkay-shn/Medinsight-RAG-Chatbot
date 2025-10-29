from datasets import load_dataset
import os

# Ensure Hugging Face library does not operate in offline mode
os.environ["HF_HUB_OFFLINE"] = "0"

print("--- Test Started ---")
print("Attempting to download the 'Laurent1/MedQuad-MedicalQnADataset_128tokens_max' dataset from Hugging Face...")

try:
    # Streaming mode ensures large datasets do not overload memory
    dataset = load_dataset(
        "Laurent1/MedQuad-MedicalQnADataset_128tokens_max",
        split="train",
        streaming=True
    )

    # Test: show first 3 examples
    print("\nSUCCESS: Dataset streaming initialized successfully!")
    print("Displaying first 3 entries for verification:")

    for i, example in enumerate(dataset):
        print(f"{i+1}: {example}")
        if i == 2:
            break

except Exception as e:
    print("\n!!! ERROR: An issue occurred while initializing dataset streaming. !!!")
    print("Detailed error message:")
    print(e)

print("\n--- Test Completed ---")