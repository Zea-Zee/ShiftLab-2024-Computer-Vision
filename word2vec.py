from gensim.models import Word2Vec
import string


english_corpus = [[char] for char in string.ascii_lowercase]
russian_corpus = [[char] for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя']

english_model = Word2Vec(english_corpus, vector_size=3, min_count=1)
russian_model = Word2Vec(russian_corpus, vector_size=10, window=1, min_count=1, sg=0)

english_vectors = {char: english_model.wv[char] for char in string.ascii_lowercase}
russian_vectors = {char: russian_model.wv[char] for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'}


print(f"English w2v vector size: {len(english_vectors['a'])}")
print(f"Russian w2v vector size: {len(russian_vectors['а'])}\n\n")


for char, vector in english_vectors.items():
    print(f"'{char}': {vector}")
