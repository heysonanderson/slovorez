from slovorez.core import cache_utils, wrapper
from slovorez.core.models import CHAR_VOCAB, UPOS, UNK_ID, morphemes_bies, morphemes_vocab
from slovorez.io import loaders

import os
from pathlib import Path


def get_project_path(relative_path):
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    full_path = project_root / relative_path
    return full_path

tikhonov_path = get_project_path("data/dictionaries/tikhonov-morphemes.json")
model_path = get_project_path("data/ml/models/resnet-deep-4b-ef-2048-80-0.4-tikh-synth.keras")

os.makedirs(tikhonov_path.parent, exist_ok=True)
os.makedirs(model_path.parent, exist_ok=True)


lex = wrapper.Lexer()
text = lex.run()


#######################

#######################


stream = cache_utils.to_token_stream(text)

# CREATING INITIAL DICT
if not os.path.exists(tikhonov_path):
    loaders.parse_tikhonov_txt()

cache = loaders.load_json(tikhonov_path)
not_in_cache, missing_count = cache_utils.find_uncached(stream, cache)


#######################

#######################


import keras
import numpy as np

from keras.utils import pad_sequences
from slovorez.ml.layers import *

# LOADING MORPHEME-SEGMENTATION MODEL
model: keras.Model = keras.models.load_model(model_path, compile=False)

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
            segments.append((word[i], morphemes_vocab[morph_type]))
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
            segments.append((seg_text, morphemes_vocab[morph_type]))
            
        else:
            morph_type = label.split('-', 1)[-1] if '-' in label else label
            if i < len(word):
                segments.append((word[i], morphemes_vocab[morph_type]))
            i += 1
    
    return segments



reverse_morphemes_bies = { v: k for k, v in morphemes_bies.items() }
reverse_morphemes_vocab = { v: k for k, v in morphemes_vocab.items() }

for word_predictions, word in zip(all_predictions[:100], not_in_cache[:100]):
    segmented_word = prediction_to_string(word, word_predictions, reverse_morphemes_bies)
    strings = [f"{seg}|({reverse_morphemes_vocab[seg_type]})" for seg, seg_type in segmented_word]
    f = "|".join(strings)
    print(f"{word} ----> {f}")


# PRINT SOME STATS
print(f"\n\n" + "-"*50 + "\n")
print(f"Full text length: {len(stream.tokens)}")
print(f"None cached words count: {missing_count}")
print(f"Uncached ratio: {missing_count / len(stream.tokens)}")
print(f"Number of new cached keys: {len(not_in_cache)}")
