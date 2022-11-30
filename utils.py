import csv
import math

## learning rate code adopted from https://d2l.ai/chapter_optimization/lr-scheduler.html
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.001, final_lr=0.0001,
                 warmup_steps=10, warmup_begin_lr=0.0001):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                        math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

def save_result(result_dic, path=''):
    with open(path + "evaluation_metrics.csv", "w") as file:
        writer = csv.DictWriter(file, result_dic.keys())
        writer.writeheader()
        writer.writerow(result_dic)

