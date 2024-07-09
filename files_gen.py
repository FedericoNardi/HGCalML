import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input folder', type=str, default='/media/disk/photon_data/v2')
parser.add_argument('--output', help='output folder', type=str, default='.')
parser.add_argument('--test', help='test fraction', type=float, default=0.1)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
args = parser.parse_args()

random.seed(args.seed)

files = os.listdir(args.input)
random.shuffle(files)

test_size = int(len(files)*args.test)

with open(args.output+'/train_files.txt', 'w') as f:
    for file in files[:-test_size]:
        print(file)
        if ".root" in file: f.write(args.input+'/'+file+'\n') 

with open(args.output+'/test_files.txt', 'w') as f:
    for file in files[-test_size:]:
        if ".root" in file: f.write(args.input+'/'+file+'\n') 

