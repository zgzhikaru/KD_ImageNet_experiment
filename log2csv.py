import re, os
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--log", default="kd/resnet18_from_resnet34.txt", required=True, 
                    help="Directory of log file to parse")
parser.add_argument("--test-only", action='store_true', 
                    help="Toggle whether to parse for only a test log")
parser.add_argument("--train-only", action='store_true', 
                    help="Toggle whether to parse for only a test log")


args = parser.parse_args()
log_path = args.log
filename = Path(log_path).with_suffix('')
#filename = os.path.basename(log_path).stem
print(f"Save as: {filename}.csv")

#file = open(args.log)
train_attr_list = [
    'Epoch',
    'sub_iter',
    'num_iter_per_epoch',
    'lr', 
    'img/s', 
    'batch_loss_med',
    'epoch_loss_avg',
    'time',
]
val_attr_list = [
    'Epoch',
    'Acc@1', 
    'Acc@5',
]
test_attr_list = [
    'type',
    'arch',
    'Acc@1', 
    'Acc@5',
]

if not args.test_only:
    train_match = re.findall(r'Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\][\w\s:]*lr:\s*([.\d]+)\s*img/s:\s*(\d+.\d+)\s*loss:\s*(\d+.\d+)\s*\((\d+.\d+)\)\s*time:\s*(\d*.\d*)', open(log_path).read())
    val_match = re.findall(r'Epoch:\s*\[(\d+)\].*?Acc@1\s*(\d+.\d+)\s*Acc@5\s*(\d+.\d+)', open(log_path).read(), re.DOTALL)

    assert len(train_match) and len(train_match[0]) == len(train_attr_list)
    assert len(val_match) and len(val_match[0]) == len(val_attr_list)

    train_logs = pd.DataFrame(train_match, columns=train_attr_list)
    val_logs = pd.DataFrame(val_match, columns=val_attr_list)

    train_logs.to_csv(f'{filename}_train.csv', index=False)
    val_logs.to_csv(f'{filename}_val.csv', index=False)

if not args.train_only:
    test_match = re.findall(r'\[(Teacher|Student):\s*(\S+)\].*?Acc@1\s*(\d+.\d+)\s*Acc@5\s*(\d+.\d+)', open(log_path).read(), re.DOTALL)
    assert len(test_match) > 0 and len(test_match) <= 2 and len(test_match[0]) == 4

    test_logs = pd.DataFrame(test_match, columns=test_attr_list)
    test_logs.to_csv(f'{filename}_test.csv', index=False)

print("done")