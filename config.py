import random

import os
import nltk


def get_texts(n=1000):
    nltk.download('words')

    word_list = nltk.corpus.words.words()
    preprocessed_words = [word for word in word_list if 4 <= len(word) <= 12]
    preprocessed_words = [word for word in word_list if 4 <= len(word) <= 12]

    sampled_words = random.sample(preprocessed_words, n * 10)
    words = [word.lower() if random.randint(0, 5) else word.upper() for word in sampled_words]
    return words[:n]


def get_fonts_paths(dir='fonts'):
    font_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.ttf'):
                font_paths.append(os.path.join(root, file))
    return font_paths


fonts_paths = get_fonts_paths()
texts = get_texts()
# print(get_words())
