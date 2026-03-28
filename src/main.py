from slovorez.core import cache_utils
from slovorez.core.models import CHAR_VOCAB, UPOS, UNK_ID, PAD_ID, morphemes_bies, morphemes_vocab
from slovorez.analytics.morphemes import parse_tikhonov_txt
from slovorez.utils import get_project_path, file_exists
from line_profiler import LineProfiler
import queue
import sys
import os
import numpy as np
from slovorez.ml.layers import *
from slovorez.core.wrapper import Sentencer
from slovorezCXX import TokenType
from slovorez.io import loaders
from torch import multiprocessing

dictionary_path = get_project_path("data/dictionaries/static_dictionary/tikhonov-morphemes-pos.json", create_dir=True)
model_path = get_project_path("data/ml/models/resnet-deep-4b-ef-2048-80-0.4-tikh-synth.keras", create_dir=True)
text_path = "large.txt"


cache = loaders.load_json(dictionary_path)
cache_set = set(cache.keys())
uncached = set()
current_uncached = set()

full_text_len = 0
missing_count = 0

#######################

#######################

#######################

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
reverse_char_vocab = { v: k for k, v in CHAR_VOCAB.items() }

# --------- MAIN DRIVER ---------

def gpu_worker(model_path, gpu_queue, result_queue, device_id=0):
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import torch
    
    lp = LineProfiler()
    
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    model = keras.models.load_model(model_path, compile=False)

    # dummy = torch.zeros((1, 64), dtype=torch.long, device=device)
    # model(dummy)

    def run_inference():
        with torch.no_grad():
            while True:
                data = gpu_queue.get()
                if data is None:
                    break

                X_batch = torch.from_numpy(data).long().to(device)

                preds = model(X_batch)
                preds = preds.half().cpu().numpy()

                result_queue.put({
                    "preds": preds,
                    "words": data
                })

    lp_wrapper = lp(run_inference)
    lp_wrapper()
    
    print(f"\n{'='*20} GPU WORKER STATS {'='*20}")
    lp.print_stats(stream=sys.stdout)

def fast_padding(tokenized_list, maxlen=64):
    current_max = max(len(t) for t in tokenized_list)
    actual_len = min(current_max, maxlen)
    arr = np.zeros((len(tokenized_list), actual_len), dtype=np.int32)
    for i, tokens in enumerate(tokenized_list):
        t_len = min(len(tokens), actual_len)
        arr[i, :t_len] = tokens[:t_len]
    return arr
        
def cpu_worker(task_queue, gpu_queue, cache_set, batch_size_limit=4096):
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    def process_data():
        local_seen = set() 
        get_char = CHAR_VOCAB.get
        
        pending_lower = []
        append_lower = pending_lower.append

        def send_to_gpu(lower_list):
            char_tokenized = [[get_char(c, UNK_ID) for c in t] for t in lower_list]
            X_input = fast_padding(char_tokenized, maxlen=64)
            gpu_queue.put(X_input)

        while True:
            batch = task_queue.get()
            if batch is None: break

            tokens = batch["text"].lower().split('\0')[:-1]

            new_tokens = set(tokens) # { t for t in tokens if len(t) > 3 and len(t) < 65}

            new_tokens = new_tokens - cache_set - local_seen

            new_tokens = sorted(new_tokens)

            for t_low in new_tokens:
                append_lower(t_low)
                local_seen.add(t_low)
                if len(pending_lower) >= batch_size_limit:
                    send_to_gpu(pending_lower)
                    pending_lower.clear()


        if pending_lower:
            send_to_gpu(pending_lower)


    lp_wrapper = lp(process_data)
    lp_wrapper()

    print(f"\n{'='*20} СPU WORKER STATS {'='*20}")
    lp.print_stats(stream=sys.stdout)

def writer_worker(result_queue, output_path):
    print("Writer worker started")
    import json

    local_cache = {}
    
    while True:
        res = result_queue.get()
        if res is None:
            break
            
        words = res["words"]
        preds = res["preds"]
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
            
        for word, p in zip(words, preds):
            word = "".join([reverse_char_vocab.get(c, UNK_ID) for c in word if c != 0])
            parsed = prediction_to_string(word, p, reverse_morphemes_bies)
            local_cache[word] = parsed

        if len(local_cache) > 8192:
            print(next(iter(local_cache.items())))
            local_cache.clear()

    print("Writer worker finished")

MODEL_THNUM = 4
NUM_CPU_WORKERS = 8
MODEL_BATCHSIZE = 2048
TASK_QUEUE_LIMIT = 16

def main():
    multiprocessing.set_start_method('spawn', force=True)
    gpu_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    task_queue = multiprocessing.Queue(TASK_QUEUE_LIMIT)

    gpu_proc = multiprocessing.Process(target=gpu_worker, args=(model_path, gpu_queue, result_queue))
    gpu_proc.start()

    writer_proc = multiprocessing.Process(target=writer_worker, args=(result_queue, "results.xml"))
    writer_proc.start()

    inactive_workers = [multiprocessing.Process(target=cpu_worker, args=(task_queue, gpu_queue, cache_set, MODEL_BATCHSIZE)) for _ in range(NUM_CPU_WORKERS)]
    active_workers = []

    if inactive_workers:
        w = inactive_workers.pop()
        w.start()
        active_workers.append(w)

    sentencer = Sentencer(text_path)
    sentencer.set_batch_size(65536) # 16777216 # 8388608 # 4194304 # 2097152 # 65536 # 131072
    sentencer.set_filter(TokenType.RUWORD)

    for batch in sentencer.stream:
        while True:
            try:

                task_queue.put(batch, block=True, timeout=0.1)
                break
            except queue.Full:
                if inactive_workers:
                    w = inactive_workers.pop()
                    w.start()
                    active_workers.append(w)
                    print(f"Added worker. Total active: {len(active_workers)}")
                else:
                    task_queue.put(batch)
                    break

    for _ in range(len(active_workers)): 
        task_queue.put(None)
    
    for w in active_workers: 
        w.join()

    gpu_queue.put(None)
    gpu_proc.join()

    result_queue.put(None)
    writer_proc.join()

    print("All processes finished successfully")

if __name__ == "__main__":
    from line_profiler import LineProfiler
    lp = LineProfiler()

    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
