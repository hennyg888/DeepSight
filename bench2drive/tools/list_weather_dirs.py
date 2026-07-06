import os
import re

ROOT = "/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted"
WEATHERS = {8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 25}

pattern = re.compile(r"Weather(\d+)$")

for name in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, name)):
        continue
    match = pattern.search(name)
    if match and int(match.group(1)) in WEATHERS:
        print(name)
