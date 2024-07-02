import json
import math

BATCH_SIZE = 6
LAST_MOE_LAYER = 11

def filter_only_encoder_and_batch(data):
    batches = []
    batch_num = 0
    cur_batch = {}
    for entry in data:
        #           Implies decoder        Only care about first layer
        if len(entry["ExpertIndices"]) == 1 or entry["Layer"] != 1: 
            continue

        if entry["Layer"] not in cur_batch:
            cur_batch[entry["Layer"]] = []
        cur_batch[entry["Layer"]].append(entry["ExpertIndices"])
        
        if len(cur_batch[entry["Layer"]]) == BATCH_SIZE:
            collapsed = []
            for seq in cur_batch[entry["Layer"]]:
                collapsed.extend(seq)
            batches.append({"Batch": batch_num, "Layer": entry["Layer"], "ExpertIndices": collapsed})
            cur_batch[entry["Layer"]] = []
            batch_num += 1
    return batches 

def get_avg_token_length_in_batch(data):
    token_nums = []
    for entry in data:
        token_nums.append(len(entry["ExpertIndices"]))
    mean = sum(token_nums) / len(token_nums)
    variance = sum((x - mean) ** 2 for x in token_nums) / (len(token_nums) - 1)
    st_div = math.sqrt(variance)
    return mean, st_div


with open("data/expert-choices-bookcorpus.json", "r") as f:
     data = json.load(f)
     data = filter_only_encoder_and_batch(data)
     print(get_avg_token_length_in_batch(data))