from torch.utils.data import Dataset, DataLoader
from util import plot_loss, str2bool, str2tuple, s2ituple
from model import model_init, BraggNN
from dataset import PatchDataset
import torch, argparse, os
import pandas as pd 
import numpy as np
import logging 
import time 
import sys 

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-ge_ffile', type=str, required=True, help='frame ge3 file')
parser.add_argument('-ge_dfile', type=str, required=True, help='frame ge3 file')
parser.add_argument('-n_frames', type=int, default=1440,  help='number of frames')
parser.add_argument('-fcsz',   type=s2ituple, default='16_8_4_2', help='size of dense layers')
parser.add_argument('-psz',    type=int, default=15, help='working patch size')
parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=500, help='max training epoches')
parser.add_argument('-exp_name',type=str, default="debug", help='Experiment name')
args, unparsed = parser.parse_known_args()

def main(args):
    if os.path.exists(args.exp_name) is False: 
        os.makedirs(args.exp_name) 

    logging.basicConfig(filename=os.path.join(args.exp_name, 'BraggNN.log'), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)

    # Create an instance of the PatchDataset class
    dataset = PatchDataset(args.ge_ffile, args.ge_dfile, nFrames=args.n_frames)

    #initiate 0.8/0.2 split for train/val sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    ds_train,ds_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create a dataloader with a batch_size of mbsz
    dl_train = DataLoader(ds_train, batch_size=args.mbsz, shuffle=True, drop_last=True)
    dl_valid = DataLoader(ds_val,   batch_size=args.mbsz, shuffle=True, drop_last=True)

    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    time_on_training = 0

    train_loss = []
    valid_loss = []
    for epoch in range(args.maxep):
        ep_tick = time.time()
        time_comp = 0
        loss = 0.0 
        for X_mb, y_mb in dl_train:
            X_mb = X_mb.float()
            y_mb = torch.reshape(y_mb,(args.mbsz,2))
            X_mb = torch.reshape(X_mb, (args.mbsz, 1, args.psz, args.psz))
            it_comp_tick = time.time()

            optimizer.zero_grad()
            pred = model.forward(X_mb.to(device))
            loss = criterion(pred, y_mb.to(device))
            loss.backward()
            optimizer.step()
            time_comp += 1000 * (time.time() - it_comp_tick)
        train_loss.append(loss.item())

        time_e2e = 1000 * (time.time() - ep_tick)
        time_on_training += time_e2e

        _prints = '[%.3f] Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms/epoch, %.2f%%)' % (\
                   time.time(), epoch, args.psz * loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e)
        logging.info(_prints)
        pred_train = pred.cpu().detach().numpy()  
        true_train = y_mb.cpu().numpy()  
        l2norm_train = np.sqrt((true_train[:,0] - pred_train[:,0])**2   + (true_train[:,1] - pred_train[:,1])**2) * args.psz

        logging.info('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels).' % (\
                     (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5))) ) )
        pred_val, gt_val = [], []
        for X_mb_val, y_mb_val in dl_valid:
            with torch.no_grad():
                X_mb_val = X_mb_val.float()
                y_mb_val = torch.reshape(y_mb_val,(args.mbsz,2))
                X_mb_val = torch.reshape(X_mb_val, (args.mbsz, 1, args.psz, args.psz))
                _pred = model.forward(X_mb_val.to(device))
                pred_val.append(_pred.cpu().numpy())
                gt_val.append(y_mb_val.numpy())
        pred_val = np.concatenate(pred_val, axis=0)
        gt_val   = np.concatenate(gt_val,   axis=0)
        l2norm_val   = np.sqrt((gt_val[:,0]     - pred_val[:,0])**2     + (gt_val[:,1]     - pred_val[:,1])**2)   * args.psz
        
        logging.info('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels) \n' % (\
                    (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5))) ) )
        
        torch.save(model, "%s/mdl-it%05d.pth" % (args.exp_name, epoch))

    plot_loss(train_loss,'Train')
    logging.info("Trained for %3d epoches, each with %d steps (BS=%d) took %.3f seconds" % (\
                 args.maxep, len(dl_train), args.mbsz, time_on_training*1e-3))



if __name__ == "__main__":
    main(args)

