from transformers import MarianMTModel, MarianTokenizer

# Function to translate reviews from English to French
def translate_to_french(reviews):
    # Load the MarianMT model and tokenizer for English to French translation
    model_name = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input reviews and generate translations
    translated = []
    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs)
        translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(translation)
    
    return translated

# English reviews
reviews = [
    "The product is fantastic! I love it.",
    "The service was terrible, very disappointed.",
    "It's okay, not great but not bad."
]

# Translate to French
translated_reviews = translate_to_french(reviews)
print(translated_reviews)
