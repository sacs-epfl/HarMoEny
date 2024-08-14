import torch 
import torch.nn as nn
import threading
import pandas as pd 

EXPERIMENT_NUMBER_OF_TIMES = 10

class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.wi = nn.Linear(768, 2048, bias=False)
        self.wo = nn.Linear(2048, 768, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()


    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


class SwitchTransformersDenseActDenseFused(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.wis = nn.Parameter(torch.rand(num_experts, 768, 2048))
        self.wos = nn.Parameter(torch.rand(num_experts, 2048, 768))
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()
    
    def forward(self, hidden_states):
        hidden_states = torch.bmm(hidden_states, self.wis)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.bmm(hidden_states, self.wos)

        return hidden_states


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

comp_event = torch.cuda.Event(enable_timing=False)
comm_in_event = torch.cuda.Event(enable_timing=False)

comm_in_stream = torch.cuda.Stream(device="cuda:0")
comp_stream = torch.cuda.Stream(device="cuda:1")
comm_out_stream = torch.cuda.Stream(device="cuda:1")

def experiment_non_fused(num_experts, num_toks):    
    non_fused = [SwitchTransformersDenseActDense() for _ in range(num_experts)]
    for idx in range(len(non_fused)):
        non_fused[idx] = non_fused[idx].to("cuda:1")

    def send_input():
        def callback(future):
            with torch.cuda.stream(comm_in_stream):
                _input = future.value().to("cuda:1", non_blocking=True)

                # Do not return the callback until finished
                comm_in_event.record()
                comm_in_event.synchronize()

                return _input
        return callback 
    
    def perform_computation(expert):
        def callback(future):
            with torch.cuda.stream(comp_stream):
                _input = future.value()
                _output = expert.forward(_input)
                
                # Do not return the callback until finished
                comp_event.record()
                comp_event.synchronize()

                return _output
        return callback

    def send_output():
        def callback(future):
            with torch.cuda.stream(comm_out_stream):
                output = future.value()
                output.to("cuda:0", non_blocking=True)
        return callback

    times = []
    for i in range(EXPERIMENT_NUMBER_OF_TIMES):
        # Setup
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        _in = [ torch.rand(num_toks, 768, device="cuda:0") for _ in range(num_experts)]
        
        
        def do_work_on_expert(expert_idx):
            future = torch.futures.Future()
            future.set_result(_in[expert_idx])
            future = future.then(send_input())
            future = future.then(perform_computation(non_fused[expert_idx]))
            future = future.then(send_output())

        start_event.record()
        threads = []
        for expert_idx in range(num_experts):
            thread = threading.Thread(target=do_work_on_expert, args=(expert_idx,))
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()

        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return sum(times) / len(times)



def experiment_bmm(num_experts, num_toks):
    fused = SwitchTransformersDenseActDenseFused(num_experts)
    fused = fused.to("cuda:1")

    def send_input():
        def callback(future):
            with torch.cuda.stream(comm_in_stream):
                _input = future.value().to("cuda:1", non_blocking=True)

                # Do not return the callback until finished
                comm_in_event.record()
                comm_in_event.synchronize()

                return _input
        return callback 
    
    def perform_computation():
        def callback(future):
            with torch.cuda.stream(comp_stream):
                _input = future.value()
                _output = fused.forward(_input)
                
                # Do not return the callback until finished
                comp_event.record()
                comp_event.synchronize()

                return _output
        return callback

    def send_output():
        def callback(future):
            with torch.cuda.stream(comm_out_stream):
                output = future.value()
                output.to("cuda:0", non_blocking=True)
        return callback

    times = []
    for i in range(EXPERIMENT_NUMBER_OF_TIMES):
        # Setup
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        _in = torch.rand(num_experts, num_toks, 768, device="cuda:0")

        start_event.record()
        future = torch.futures.Future()
        future.set_result(_in)
        future = future.then(send_input())
        future = future.then(perform_computation())
        future = future.then(send_output())
        end_event.record()
        
        future.wait()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return sum(times) / len(times)


torch.cuda.cudart().cudaProfilerStart()
EXPERTS_TO_TEST = [1, 2, 4, 8, 16]
TOKENS_TO_TEST = [1, 10, 100, 1000, 10000, 100000]

results = []

for i in EXPERTS_TO_TEST:
    for j in TOKENS_TO_TEST:
        if i == 16 and j == 100000:
            continue
        print(f"({i},{j})")
        results.append({
            "num_experts": i,
            "num_tokens": j,
            "avg_time_non_fused": experiment_non_fused(i, j),
            "avg_time_bmm": experiment_bmm(i, j)
        })

df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('exp1_results.csv', index=False)

print("Results saved to exp1_results.csv")

torch.cuda.cudart().cudaProfilerStop()




