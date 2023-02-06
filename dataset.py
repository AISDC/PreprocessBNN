import os 
import h5py 
import torch
import numpy as np
from numba import jit
from skimage import measure
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.measure import label, regionprops

@jit
def process_frame(frame, dark, thresh):
    frame = frame - dark
    frame[frame < thresh] = 0
    frame = frame.astype(int)
    return frame

def normalize_patch(patch):
    patch = patch.astype(float)
    _min,_max = patch.min().astype(np.float32), patch.max().astype(np.float32)
    feature   = (patch - _min) / (_max- _min) 
    return feature


class PatchDataset(Dataset): 
    def __init__(self, ffile, dfile, nFrames, NrPixels=2048, thresh=100, fHead=8192, window=7):
        self.NrPixels = NrPixels

        # Read dark frame
        with open(dfile, 'rb') as darkf:
            darkf.seek(fHead+NrPixels*NrPixels*2, os.SEEK_SET)
            self.dark = np.fromfile(darkf, dtype=np.uint16, count=(NrPixels*NrPixels))
            self.dark = np.reshape(self.dark,(NrPixels,NrPixels))
            self.dark = self.dark.astype(float)
        darkf.close()

        # Read frames
        self.frames = []
        self.length = 0
        self.xy_positions = []
        self.patches=[]
        self.f_nums=[]
        self.p_nums=[]
        with open(ffile, 'rb') as f:
            for fNr in range(1,nFrames+1):
                BytesToSkip = fHead + fNr*NrPixels*NrPixels*2
                f.seek(BytesToSkip,os.SEEK_SET)
                thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
                thisFrame = np.reshape(thisFrame,(NrPixels,NrPixels))
                thisFrame = thisFrame.astype(float)
                thisFrame = process_frame(thisFrame, self.dark, thresh)
                thisFrame2 = np.copy(thisFrame)
                thisFrame2[thisFrame2>0] = 1
                labels = label(thisFrame2)
                regions = regionprops(labels)                                                       
                i=1
                for prop_nr,props in enumerate(regions):
                    if props.area < 4 or props.area > 150:
                        continue
                    y0,x0   = props.centroid
                    start_x = int(x0)-window
                    end_x   = int(x0)+window+1
                    start_y = int(y0)-window
                    end_y   = int(y0)+window+1
                    if start_x < 0 or end_x > NrPixels - 1 or start_y < 0 or end_y > NrPixels - 1:
                        continue
                    sub_img = thisFrame[start_y:end_y,start_x:end_x]
                    self.patches.append(normalize_patch(sub_img))
                    self.xy_positions.append([start_y,start_x])
                    self.f_nums.append(fNr)
                    self.p_nums.append(i)
                    i+=1 
                self.length = len(self.patches)
        f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.patches[index], self.xy_positions[index], self.f_nums[index], self.p_nums[index]  

