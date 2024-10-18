import unittest
import torch
import random
from scheduler import Scheduler 

class TestScheduler(unittest.TestCase):
    def test_basic(self):
        scheduler = Scheduler(num_experts=2, num_gpus=2, scheduling_policy="adfabricus")
        meta = [[torch.tensor(10), torch.tensor(0)], [torch.tensor(0), torch.tensor(20)]]
        topo = [[1], [0]]
        schedule = scheduler(meta, topo)
        expected_schedule = [[[0, 10], [0, 0]], [[0, 0], [20, 0]]]
        self.assertEqual(schedule, expected_schedule) 
    
    def test_all_tokens_scheduled(self):
        num_experts=8
        num_gpus=4

        scheduler = Scheduler(num_experts=num_experts, num_gpus=num_gpus, scheduling_policy="adfabricus")
        topo = [[0,1], [2,3], [4,5], [6,7]]

        # Just do it 10 times
        for _ in range(10):
            random.seed(42)
            meta = [[torch.tensor(random.randint(0, 1000)) for _ in range(num_experts)] for _ in range(num_gpus)]
            schedule = scheduler(meta, topo)
            num_tot_tokens = sum(list(map(lambda arr: sum(list(map(lambda x: x.item(), arr))), meta)))
            num_scheduled_tokens = sum(list(map(lambda experts: sum(list(map(lambda t: sum(t), experts))), schedule)))
            self.assertEqual(num_scheduled_tokens, num_tot_tokens)

if __name__ == "__main__":
    unittest.main()