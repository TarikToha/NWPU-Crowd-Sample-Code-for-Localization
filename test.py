from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

import pdb

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '../dataset/nwpu/min_576x768_mod16_2048'
ori_data = '../dataset/nwpu/nwpu_val'  # get the original size
model_path = 'exp/04-25_05-19_NWPU_RAZ_loc_1e-05/all_ep_125_bceloss_0.065320.pth'
# model_path = 'exp/05-04_23-54_NWPU_LC_Net_1e-05/all_ep_875_bceloss_0.066730.pth

def main():
    txtpath = os.path.join(dataRoot, 'val.txt')
    with open(txtpath) as f:
        lines = f.readlines()
    test(lines, model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, 'RAZ_loc')
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    gts = []
    preds = []

    record = open('RAZ_loc_out.txt', 'w+')
    for infos in file_list:
        filename = infos.split()[0]

        imgname = os.path.join(dataRoot, 'img', filename + '.jpg')
        img = Image.open(imgname)
        ori_img = Image.open(os.path.join(ori_data, filename + '.jpg'))
        ori_w, ori_h = ori_img.size
        w, h = img.size

        ratio_w = ori_w / w
        ratio_h = ori_h / h

        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        with torch.no_grad():
            img = Variable(img).cuda()
            crop_imgs, crop_masks = [], []
            b, c, h, w = img.shape
            rh, rw = 576, 768
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros(b, 1, h, w).cuda()
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            # forward may need repeatng
            crop_preds = []
            nz, bz = crop_imgs.size(0), 1
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i + bz)
                crop_pred = net.test_forward(crop_imgs[gs:gt])

                crop_pred = F.softmax(crop_pred, dim=1).data[0, 1, :, :]
                crop_pred = crop_pred[None, :, :]

                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros(b, 1, h, w).cuda()
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1

            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            pred_map = pred_map / mask

        pred_map = F.avg_pool2d(pred_map, 3, 1, 1)
        maxm = F.max_pool2d(pred_map, 3, 1, 1)
        maxm = torch.eq(maxm, pred_map)
        pred_map = maxm * pred_map
        pred_map[pred_map < 0.5] = 0
        pred_map = pred_map.bool().long()
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        ids = np.array(np.where(pred_map == 1))  # y,x
        ori_ids_y = ids[0, :] * ratio_h
        ori_ids_x = ids[1, :] * ratio_w
        ids = np.vstack((ori_ids_x, ori_ids_y)).astype(np.int16)  # x,y

        loc_str = ''
        for i_id in range(ids.shape[1]):
            loc_str = loc_str + ' ' + str(ids[0][i_id]) + ' ' + str(ids[1][i_id])  # x, y

        pred = ids.shape[1]

        print(f'{filename} {pred:d}{loc_str}', file=record)
        print(f'{filename} {pred:d}')
    record.close()


if __name__ == '__main__':
    main()
