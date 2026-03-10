from slovorez.core import cache_utils
from slovorez.core.models import CHAR_VOCAB, UPOS, UNK_ID, morphemes_bies, morphemes_vocab
from slovorez.analytics.morphemes import parse_tikhonov_txt
from slovorez.utils import get_project_path, file_exists

dictionary_path = get_project_path("data/dictionaries/ml-morphemes.json", create_dir=True)
model_path = get_project_path("data/ml/models/resnet-deep-4b-ef-2048-80-0.4-tikh-synth.keras", create_dir=True)
text_path = "large.txt"

import gc
import psutil
import os
from slovorez.core.wrapper import Sentencer


sentencer = Sentencer(text_path)
sentencer.set_batch_size(65536)


# CREATING INITIAL DICT
if not file_exists(dictionary_path):
    parse_tikhonov_txt()
cache = {} # loaders.load_json(dictionary_path)
cache_manager = set(cache.keys())
uncached = set()
current_uncached = set()

full_text_len = 0
batch_count = 0
missing_count = 0


proc = psutil.Process(os.getpid())


def iterate_batches(sentencer):
    while(True):
        batch = sentencer.get_batch()
        if not batch:
            break
        
        batch_count = batch_count + 1
        full_text_len+=len(batch)
    
def check_uncached(current_batch):
    return cache_utils.find_uncached(stream=current_batch, cache_set=cache_manager, uncached=uncached)


uncached = list(uncached)
print(uncached)
#######################

#######################

#######################

#######################


import keras
import numpy as np

from keras.utils import pad_sequences
from slovorez.ml.layers import *

# LOADING MORPHEME-SEGMENTATION MODEL
model: keras.Model = keras.models.load_model(model_path, compile=False)

# CREATING PADDED SEQUENCES
X_input = pad_sequences(char_tokenized, maxlen=64, padding='post', value=0)
predictions = model.predict(X_input, batch_size=1024, verbose=1)
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

for word_predictions, word in zip(all_predictions[:100], uncached[:100]):
    segmented_word = prediction_to_string(word, word_predictions, reverse_morphemes_bies)
    strings = [f"{seg}|({reverse_morphemes_vocab[seg_type]})" for seg, seg_type in segmented_word]
    f = "|".join(strings)
    print(f"{word} ----> {f}")


# PRINT SOME STATS
print(f"\n\n" + "-"*50 + "\n")
print(f"Full text length: {full_text_len}")
print(f"None cached words count: {missing_count}")
print(f"Uncached ratio: {missing_count / full_text_len}")
print(f"Number of new cached keys: {len(uncached)}")
