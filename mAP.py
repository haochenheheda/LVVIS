from evaluate.lvvis import LVVIS
from evaluate.lvviseval import LVVISeval
import sys
import numpy as np


gt_path = '/home/haochen/workspace/datasets/VIS/LVVIS/val/val_instances.json'
dt_path = 'output/prompt3/inference/lvvis_val/results.json'

ytvosGT = LVVIS(gt_path)
ytvosDT = ytvosGT.loadRes(dt_path)
ytvosEval = LVVISeval(ytvosGT, ytvosDT, "segm")
ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()


