import json
import csv
import math

NUM_EXPERTS = 8
NUM_GPUS = 4
GPU_ASSIGNMENT_NAIVE = [[0,1],[2,3],[4,5],[6,7]]
GPU_ASSIGNMENT_DEMETER = [[0,1,2], [3,4,5], [6,7]]
BATCH_SIZE = 6
LAST_MOE_LAYER = 11

def calculate_expert_freq(data):
    freq = []
    def get_freq(assignment):
        experts = [0] * NUM_EXPERTS
        for expert in assignment:
            experts[expert] += 1
        return experts

    for assignment in data:
        _freq = []
        if isinstance(assignment["ExpertIndices"], list) and isinstance(assignment["ExpertIndices"][0], list):
            for partition in assignment["ExpertIndices"]:
                _freq.append(get_freq(partition))  
        else:
            _freq = get_freq(assignment["ExpertIndices"])
        freq.append({"Batch": assignment["Batch"], "Layer": assignment["Layer"], "ExpertFreq": _freq})
            
    return freq

def calculate_penalty(assignment):
    penalties = []
    for batch in assignment:
        _sum = 0
        _max = 0
        for gpu_freq in batch["Assignment"]:
            _sum += gpu_freq
            if gpu_freq > _max:
                _max = gpu_freq
        avg_tokens_per_gpu = _sum / len(batch["Assignment"])
        penalty = round(_max - avg_tokens_per_gpu, 2)
        penalties.append({"Batch": batch["Batch"], "Layer": batch["Layer"], "Penalty": penalty})
    return penalties

def assign_naive(data):
    assignments = []
    for batch in data:
        gpus = [0] * len(GPU_ASSIGNMENT_NAIVE)
        for idx_e, expert in enumerate(batch["ExpertFreq"]):
            for idx_g, gpu in enumerate(GPU_ASSIGNMENT_NAIVE):
                if idx_e in gpu:
                    gpus[idx_g] += expert
        assignments.append({"Batch": batch["Batch"], "Layer": batch["Layer"], "Assignment": gpus})
    return assignments

def assign_ideal_greedy(data):
    assignments = []
    for batch in data:
        gpus = [0] * NUM_GPUS
        expert_freq = {}
        for idx_e, freq in enumerate(batch["ExpertFreq"]):
            expert_freq[idx_e] = freq
        expert_freq_sorted = dict(sorted(expert_freq.items(), key=lambda x: x[1]))
        expert_freq_sorted_list = list(expert_freq_sorted.items())
        while len(expert_freq_sorted_list) != 0:
            # Find minimal GPU
            _min = gpus[0]
            min_idx = 0
            for idx_g, count in enumerate(gpus):
                if count < _min:
                    _min = count
                    min_idx = idx_g
            gpus[min_idx] += expert_freq_sorted_list[-1][1]
            expert_freq_sorted_list.pop(-1)
        assignments.append({"Batch": batch["Batch"], "Layer": batch["Layer"], "Assignment": gpus})
    return assignments

def assign_demeter(data):
    assignments = []
    for batch in data:
        gpu_orig = [0] * len(GPU_ASSIGNMENT_DEMETER)
        _sum = 0
        for idx_e, freq in enumerate(batch["ExpertFreq"]):
            for idx_g, experts in enumerate(GPU_ASSIGNMENT_DEMETER):
                if idx_e in experts:
                    gpu_orig[idx_g] += freq
                    _sum += freq
        # Balancing
        avg_tokens_per_gpu = _sum / NUM_GPUS
        gpus = [0] * NUM_GPUS
        threshold = math.ceil(avg_tokens_per_gpu * 1.15) # POLICY
        for idx, amt in enumerate(gpu_orig):
            if amt > threshold:
                reduction = amt - threshold
                gpus[idx] = threshold
                gpus[-1] += reduction 
            else:
                gpus[idx] = amt
        assignments.append({"Batch": batch["Batch"], "Layer": batch["Layer"], "Assignment": gpus})
    
    return assignments 

def assign_demeter_dp(data):
    assignments = []
    
    for batch in data:
        gpus = [0] * NUM_GPUS
        for part in batch["ExpertFreq"]:
            gpu_orig = [0] * len(GPU_ASSIGNMENT_DEMETER)
            _sum = 0
            for idx_e, freq in enumerate(part):
                for idx_g, experts in enumerate(GPU_ASSIGNMENT_DEMETER):
                    if idx_e in experts:
                        gpu_orig[idx_g] += freq
                        _sum += freq 
            # Balancing
            avg_tokens_per_gpu = _sum / NUM_GPUS
            threshold = math.ceil(avg_tokens_per_gpu * 1.15) # POLICY
            for idx, amt in enumerate(gpu_orig):
                if amt > threshold:
                    reduction = amt - threshold
                    gpus[idx] += threshold
                    gpus[-1] += reduction 
                else:
                    gpus[idx] += amt
        assignments.append({"Batch": batch["Batch"], "Layer": batch["Layer"], "Assignment": gpus})

    return assignments

def filter_only_encoder_and_batch(data, dp_size=1):
    batches = []
    batch_num = 0
    cur_batch = {}
    for entry in data:
        if len(entry["ExpertIndices"]) == 1: # Implies decoder
            continue

        if entry["Layer"] not in cur_batch:
            cur_batch[entry["Layer"]] = []
        cur_batch[entry["Layer"]].append(entry["ExpertIndices"])
        
        if len(cur_batch[entry["Layer"]]) == BATCH_SIZE:
            collapsed = []
            cur = []
            num = 0
            target = BATCH_SIZE / dp_size
            for seq in cur_batch[entry["Layer"]]:
                cur.extend(seq)
                num += 1
                if num == target:
                    collapsed.append(cur)
                    cur = []
                    num = 0
            batches.append({"Batch": batch_num, "Layer": entry["Layer"], "ExpertIndices": collapsed})
            cur_batch[entry["Layer"]] = []
            if entry["Layer"] == LAST_MOE_LAYER:
                batch_num += 1
    return batches 

def save_penalties_to_csv(penalties, name="tmp"):
    # Reshape the dict
    reshaped_penalties = {}
    for penalty in penalties:
        if penalty["Batch"] not in reshaped_penalties:
            reshaped_penalties[penalty["Batch"]] = {}
        reshaped_penalties[penalty["Batch"]][penalty["Layer"]] = penalty["Penalty"]
    reshaped_penalties_list = []
    for batch_num, penals in reshaped_penalties.items():
        novel = {}
        novel["Batch Number"] = batch_num
        for layer_idx, penalty in penals.items():
            novel[f"Layer {layer_idx}"] = penalty
        reshaped_penalties_list.append(novel)

    with open(f"outputs/{name}.csv", mode="w", newline='') as f:
        fieldNames = ["Batch Number", "Layer 1", "Layer 3", "Layer 5", "Layer 7", "Layer 9", "Layer 11"]
        writer = csv.DictWriter(f, fieldNames)

        writer.writeheader()
        writer.writerows(reshaped_penalties_list)

        print(f"Data has been written to {name}")


with open("data/expert-choices-bookcorpus.json", "r") as f:
    data = json.load(f)
    # data = filter_only_encoder_and_batch(data)
    # freq = calculate_expert_freq(data)
    # assignment_naive = assign_naive(freq)
    # assignment_ideal_greedy = assign_ideal_greedy(freq)
    # assignment_demeter = assign_demeter(freq)
    # penalties_naive = calculate_penalty(assignment_naive)
    # penalties_ideal_greedy = calculate_penalty(assignment_ideal_greedy)
    # penalties_demeter = calculate_penalty(assignment_demeter)
    # save_penalties_to_csv(penalties_naive, "naive")
    # save_penalties_to_csv(penalties_ideal_greedy, "ideal-greedy")
    # save_penalties_to_csv(penalties_demeter, "demeter")
    data_dp_3 = filter_only_encoder_and_batch(data, dp_size=3)
    freq = calculate_expert_freq(data_dp_3)
    assignment_demeter_dp_3 = assign_demeter_dp(freq)
    penalties_demeter_dp_3 = calculate_penalty(assignment_demeter_dp_3)
    save_penalties_to_csv(penalties_demeter_dp_3, "demeter-dp-3")