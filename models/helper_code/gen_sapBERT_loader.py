from transformers import AutoTokenizer, AutoModel

model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
save_directory = "../helper_code/sapBERT_local_save" 

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"Saving to {save_directory}...")
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print("Done! The model is now ready for offline use.")