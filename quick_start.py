import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import time


model = 'meta/llama-3.1-8b-instruct'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
start = time.time()
task = Game24Task()
ys, infos = solve(args, task, 900)
lapse = time.time() - start
print(f'Time lapse of {model} is {lapse}')
print(ys[0])