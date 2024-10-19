import numpy as np
import string


english_alphabet = list(string.ascii_lowercase)
russian_alphabet = ''.join([chr(i) for i in range(ord('а'), ord('а') + 32)])

one_hot_english = np.eye(len(english_alphabet))
one_hot_russian = np.eye(len(russian_alphabet))

print(f"onehot shape: {one_hot_english[1].shape}")
for letter, encoding in zip(english_alphabet, one_hot_english):
    print(f"'{letter}': {encoding}")
    
print(f"Russian onehot shape: {one_hot_english[1].shape}")
for letter, encoding in zip(russian_alphabet, one_hot_russian):
    print(f"'{letter}': {encoding}")
