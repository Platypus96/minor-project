import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from pipeline.translation import translate_to_english, translate_from_english

print("Testing multilingual translation...")

try:
    # Test 1: Hindi to English
    hindi_text = "नमस्ते, आप कैसे हैं?"
    print(f"Original (Hindi): {hindi_text}")
    english_translation = translate_to_english(hindi_text, "hi")
    print(f"Translated to English: {english_translation}")

    # Test 2: English to Spanish
    es_translation = translate_from_english(english_translation, "es")
    print(f"Translated to Spanish: {es_translation}")
    
    print("✅ Multilingual test complete and successful.")
except Exception as e:
    print(f"❌ Error during translation: {e}")
