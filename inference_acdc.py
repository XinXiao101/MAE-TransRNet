import glob
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from tools import utils,losses
import shutil
import sys
from torch.utils.data import DataLoader
from data_pre import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
# from ignite.contrib.handlers import ProgressBar
from torchsummary import summary
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_reg
from natsort import natsorted
import pdb
import SimpleITK as sitk

result_dir = '/home/xiaoxin/MAE_TransRNet/tmp/pycharm_project_858/vis'
def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(result_dir, name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def main():
    class_num = 4
    test_dir = './npdata/testing'
    model_idx = -1
    model_folder = 'MAE_TransRNet_reg/'
    model_dir = './experiments/'
    # model_dir = './experiments/' + model_folder
    config_vit = CONFIGS_reg['MAE_TransRNet']
    dict = utils.process_label()
    model = models.MAE_TransRNet(config_vit, img_size=(64, 128, 128))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((64, 128, 128), 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.CardiacInferDataset(test_dir,transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = [AverageMeter() for i in range(class_num)]
    eval_dsc_raw = [AverageMeter() for i in range(class_num)]
    eval_det = AverageMeter()
    eval_hd = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0] # (1,1,64,128,128)
            y = data[1] # (1,1,64,128,128)
            x_seg = data[2] # (1, 1, 4, 64, 128, 128)
            y_seg = data[3] # (1, 1, 4, 64, 128, 128)
            x_in = torch.cat((x,y),dim=1)   # (1,2,64,128,128)
            x_def, flow = model(x_in)   # x_def:(1,1,64,128,128), flow: (1,3,64,128,128)
            def_out = reg_model([x_seg[:, 0, ...].cuda().float(), flow.cuda()]) # def_out:(1,4,64,128,128)
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]   # (64, 128, 128)
            jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg[:, 0, ...].long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            hd_set = utils.HD(def_out.long(), y_seg[:, 0, ...].long())
            dsc_trans = utils.dice_val_2(def_out.long(), y_seg[:, 0, ...].long()).detach().cpu().numpy()   
            dsc_raw = utils.dice_val_2(x_seg[:, 0, ...].long(), y_seg[:, 0, ...].long()).detach().cpu().numpy()
            for class_i in range(class_num):
                print('class ID {}, HD: {:.4f}, Trans diff: {:.4f}, Raw diff: {:.4f}'.format(class_i, hd_set[class_i], dsc_trans[class_i],dsc_raw[class_i]))
                eval_dsc_def[class_i].update(dsc_trans[class_i], x.size(0))
                eval_dsc_raw[class_i].update(dsc_raw[class_i], x.size(0))
                stdy_idx += 1
            eval_hd.update(hd_set[-1])
            print('global HD:{:.4f}'.format(eval_hd.avg))
            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            y_def, flow = model(y_in)
            def_out = reg_model([y_seg[:, 0, ...].cuda().float(), flow.cuda()])
            tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), x_seg[:, 0, ...].long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            out = def_out.detach().cpu().numpy()[0, 0, :, :, :]

            # save_image(x_def, x.cpu().numpy(), "warped.nii.gz")
            # save_image(flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], x, "flow.nii.gz")
            # save_image(def_out, x_seg, "label.nii.gz")


            print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            hd_set = utils.HD(def_out.long(), y_seg[:, 0, ...].long())
            dsc_trans = utils.dice_val_2(def_out.long(), x_seg[:, 0, ...].long()).detach().cpu().numpy()
            dsc_raw = utils.dice_val_2(y_seg[:, 0, ...].long(), x_seg[:, 0, ...].long()).detach().cpu().numpy()
            for class_i in range(class_num):
                print('class ID {}, HD: {:.4f}, Trans diff: {:.4f}, Raw diff: {:.4f}'.format(class_i, hd_set[class_i], dsc_trans[class_i],dsc_raw[class_i]))
                eval_dsc_def[class_i].update(dsc_trans[class_i], x.size(0))
                eval_dsc_raw[class_i].update(dsc_raw[class_i], x.size(0))
                # np.save('/home/xiaoxin/MAE_TransRNet/vis/Out_{}'.format(stdy_idx), out)
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/jac_det_{}".format(stdy_idx),jac_det)
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/flow_{}".format(stdy_idx), flow.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/x_{}".format(stdy_idx), x.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/y_{}".format(stdy_idx), y.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/x_seg_{}".format(stdy_idx), x_seg.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/y_seg_{}".format(stdy_idx), y_seg.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/x_def_{}".format(stdy_idx), x_def.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/y_def_{}".format(stdy_idx), y_def.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/def_out_{}".format(stdy_idx), def_out.detach().cpu().numpy())
                # np.save("/home/xiaoxin/MAE_TransRNet/vis/tar_{}".format(stdy_idx), tar)
                stdy_idx += 1
            eval_hd.update(hd_set[-1])
            print('global HD:{:.4f}'.format(eval_hd.avg))
        for class_i in range(class_num):
            print('class ID {}, Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(
                                                                                            class_i, 
                                                                                            eval_dsc_def[class_i].avg,
                                                                                            eval_dsc_def[class_i].std,
                                                                                            eval_dsc_raw[class_i].avg,
                                                                                            eval_dsc_raw[class_i].std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

if __name__ == '__main__':
    main()