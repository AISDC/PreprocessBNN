from torch.utils.data import Dataset, DataLoader
from dataset import PatchDataset
import torch, argparse, os
import numpy as np
import pandas as pd 

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-ge_ffile', type=str, default="debug", help='frame ge3 file')
parser.add_argument('-ge_dfile', type=str, default="debug", help='frame ge3 file')
args, unparsed = parser.parse_known_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the PatchDataset class
    dataset = PatchDataset(args.ge_ffile, args.ge_dfile, nFrames=1440)
    
    # Create a dataloader with a batch_size of 1
    bs=1
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    #you can insert your training loop here...

if __name__ == "__main__":
    main(args)

