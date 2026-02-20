from slovorez.core import cache_utils, wrapper
from slovorez.core.models import CHAR_VOCAB, UPOS, UNK_ID, morphemes_bies
from slovorez.io import loaders

import os

lex = wrapper.Lexer()
text = lex.run()


#######################

#######################


stream = cache_utils.to_token_stream(text)

# CREATING INITIAL DICT
if not os.path.exists("./data/dictionaries/tikhonov-morphemes.json"):
    loaders.parse_tikhonov_txt()

cache = loaders.load_json("./data/dictionaries/tikhonov-morphemes.json")
not_in_cache, missing_count = cache_utils.find_uncached(stream, cache)


#######################

#######################


import keras
import numpy as np

from keras.utils import pad_sequences
from slovorez.ml.layers import *

# LOADING MORPHEME-SEGMENTATION MODEL
model: keras.Model = keras.models.load_model("./data/ml/models/resnet-deep-4b-ef-2048-80-0.4-tikh-synth.keras", compile=False)

# PROCESSING CHAR TOKENS
char_tokenized = [[CHAR_VOCAB.get(c, UNK_ID) for c in token] for token in not_in_cache]

# CREATING PADDED SEQUENCES
X_input = pad_sequences(char_tokenized, maxlen=64, padding='post', value=0)
predictions = model.predict(X_input, batch_size=256, verbose=1)
all_predictions = np.argmax(predictions, axis=-1)


#######################

#######################


# CONVERT PREDICTIONS TO STRING-MORPHEMETYPE PAIRS
def prediction_to_string(word, word_predictions, reverse_morphemes_bies):
    segments = []

    i = 0
    n = len(word)


    while i < n:
        cls_id = word_predictions[i]
        label = reverse_morphemes_bies.get(cls_id, "<UNK>")

        if label == "<PAD>":
            break

        if i >= len(word):
            break

        if label.startswith("S-"):
            morph_type = label[2:] if '-' in label else label
            segments.append(f"{word[i]}({morph_type})")
            i += 1
            
        elif label.startswith("B-"):
            morph_type = label[2:] if '-' in label else label
            start = i
            i += 1

            while i < n:
                next_id = word_predictions[i]
                next_label = reverse_morphemes_bies.get(next_id, "")

                if next_label == "<PAD>":
                    break

                if next_label == f"E-{morph_type}":
                    i += 1
                    break
                elif next_label == f"I-{morph_type}":
                    i += 1
                else:
                    break

            end_idx = min(i, len(word))
            seg_text = ''.join(word[start:end_idx])
            segments.append(f"{seg_text}({morph_type})")
            
        else:
            morph_type = label.split('-', 1)[-1] if '-' in label else label
            if i < len(word):
                segments.append(f"{word[i]}({morph_type})")
            i += 1
    
    return '|'.join(segments)



reverse_morphemes_bies = { v: k for k, v in morphemes_bies.items() }

for word_predictions, word in zip(all_predictions[:100], not_in_cache[:100]):
    segmented_word = prediction_to_string(word, word_predictions, reverse_morphemes_bies)
    print(f"{word} ----> {segmented_word}")


# PRINT SOME STATS
print(f"\n\n" + "-"*50 + "\n")
print(f"Full text length: {len(stream.tokens)}")
print(f"None cached words count: {missing_count}")
print(f"Uncached ratio: {missing_count / len(stream.tokens)}")
print(f"Number of new cached keys: {len(not_in_cache)}")
