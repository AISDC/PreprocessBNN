from torch.utils.data import Dataset, DataLoader
from dataset import PatchDataset
import torch, argparse, os
import numpy as np
import pandas as pd 

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-ge_ffile', type=str, default="debug", help='frame ge3 file')
parser.add_argument('-ge_dfile', type=str, default="debug", help='frame ge3 file')
parser.add_argument('-m_file', type=str, default="debug", help='pt model file')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
args, unparsed = parser.parse_known_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        model=torch.load(args.m_file,map_location=torch.device('cpu'))
    else: 
        model=torch.load(args.m_file)
    model.eval()

    # Create an instance of the PatchDataset class
    dataset = PatchDataset(args.ge_ffile, args.ge_dfile, nFrames=1440)
    
    # Create a dataloader with a batch_size of 1
    bs=1 
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    if os.path.exists("results_%s" %args.expName) is False: 
        os.makedirs("results_%s" %args.expName)

    # Lists to hold stuff  
    x_list   = []
    y_list   = []

    x_corner = []
    y_corner = []

    fnums    = []
    pnums    = []

    dataframe=pd.DataFrame()

    with torch.no_grad():
        for patch, xy_orig, f_num, p_num in dataloader:
            patch = patch.float()
            patch = torch.reshape(patch,(bs,1,15,15))

            y_pred = model(patch)
            y_pred = y_pred.cpu().numpy()*15 

            x_list.append(y_pred[0][0])
            y_list.append(y_pred[0][1])

            x_corner.append(xy_orig[0].numpy()[0])
            y_corner.append(xy_orig[1].numpy()[0])

            fnums.append(f_num.numpy()[0])
            pnums.append(p_num.numpy()[0])

    dataframe['peakNr']         = pnums  
    dataframe['frameNr']        = fnums  
    dataframe['BNN_horPos']     = x_list 
    dataframe['BNN_vertPos']    = y_list 
    dataframe['label_centroid_horPos']  = x_corner 
    dataframe['label_centroid_vertPos'] = y_corner

    dataframe.to_csv('results_%s/midas_reconstruction_data_%s.csv' %(args.expName,args.expName))

if __name__ == "__main__":
    main(args)

