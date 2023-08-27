from evaluate.lvvis import LVVIS
from evaluate.lvviseval import LVVISeval
import sys
import numpy as np


gt_path = 'val/val_instances.json'
dt_path = 'output/ov2seg/inference/lvvis_val/results.json'

ytvosGT = LVVIS(gt_path)
ytvosDT = ytvosGT.loadRes(dt_path)
ytvosEval = LVVISeval(ytvosGT, ytvosDT, "segm")
ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()


