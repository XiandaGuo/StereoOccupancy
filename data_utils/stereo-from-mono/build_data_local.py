import os
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
from time import sleep

import PIL.Image as pil
import cv2
import numpy as np
from torch.utils.data import DataLoader

from utils import readlines
# from mscoco_dataset import *


def save_img(savename, img):
    # im = pil.fromarray(img)
    im = pil.fromarray(np.uint8(img))
    im.save(savename)


def process(i, bz, leftimages, rightimages, dispimages, save_root, left_image_names=None):
    for j in range(bz):
        if left_image_names is None:
            left_savename = os.path.join(save_root, '{:0>6d}left.png'.format(i * bz + j))
            right_savename = os.path.join(save_root, '{:0>6d}right.png'.format(i * bz + j))
            disp_savename = os.path.join(save_root, '{:0>6d}disp.png'.format(i * bz + j))
        else:
            filename = left_image_names[j]
            left_savename = os.path.join(save_root, filename.split('/')[-1][:-4]+'_left.png')
            right_savename = os.path.join(save_root, filename.split('/')[-1][:-4]+'_right.png')
            disp_savename = os.path.join(save_root, filename.split('/')[-1][:-4]+'_disp.png')

        leftimage = leftimages[j] * 255
        leftimage = np.transpose(leftimage, (1, 2, 0))
        rightimage = rightimages[j] * 255
        rightimage = np.transpose(rightimage, (1, 2, 0))

        # save left and right image
        save_img(left_savename, leftimage)
        save_img(right_savename, rightimage)

        # save disparity
        im = pil.fromarray((dispimages[j] * 256).astype('uint16'))
        im.save(disp_savename)
        if (i*bz+j)%100 == 0:
            print(i*bz+j)

def build_stero(mscoco, save_root, flg_names=False):
    _mscoco = DataLoader(mscoco, num_workers=32, batch_size=32)

    pool = Pool(48)
    results = list()

    for i, batch in enumerate(_mscoco):
        # print(i)
        leftimages = batch['image'].numpy()
        rightimages = batch['stereo_image']
        dispimages = batch['disparity'].numpy()

        bz = leftimages.shape[0]

        if flg_names:
            left_image_names = batch['left_image_name']
            l = pool.apply_async(
                process,
                args=(i, bz, leftimages, rightimages, dispimages, save_root, left_image_names))
        else:
            l = pool.apply_async(
                process,
                args=(i, bz, leftimages, rightimages, dispimages, save_root))

        results.append(l)

        sleep(0.0001)

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()

if __name__ == '__main__':
    png_path = '/mnt/cfs/algorithm/public_data/stereo/DrivingStereo/train_left/2018-10-16-11-13-47_2018-10-16-11-22-06-712.jpg'
    # png_path = '/mnt/cfs/algorithm/public_data/stereo/stereo_from_mono/DrivingStereo/midas_depths/train_left/2018-07-09-16-11-56_2018-07-09-16-11-56-502.png'
    img=cv2.imread(png_path)
    print(img.shape)

    from base_dataset import pil_loader
    im = pil_loader(png_path)
    print(np.array(im).shape)