# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Stereo-from-mono licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
from utils import readlines
# from __future__ import absolute_import, division, print_function

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil
# from .warp_dataset import WarpDataset
from warp_dataset import WarpDataset

import cv2

cv2.setNumThreads(0)


class DrivingstereoDataset(WarpDataset):

    def __init__(self,
                 data_path,
                 bg_dataset_path,
                 filenames,
                 background_filenames,
                 disparity_from_MiDas,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True,
                 disable_synthetic_augmentation=False,
                 disable_sharpening=False,
                 monodepth_model='midas',
                 disable_background=False,
                 **kwargs):

        super(DrivingstereoDataset, self).__init__(data_path, filenames, feed_height, feed_width,
                                            max_disparity,
                                            is_train=is_train, has_gt=True,
                                            disable_normalisation=disable_normalisation,
                                            keep_aspect_ratio=keep_aspect_ratio,
                                            disable_synthetic_augmentation=
                                            disable_synthetic_augmentation,
                                            disable_sharpening=disable_sharpening,
                                            monodepth_model=monodepth_model,
                                            disable_background=disable_background)

        if self.monodepth_model == 'midas':
            self.disparity_path = 'midas_depths'
        elif self.monodepth_model == 'megadepth':
            self.disparity_path = 'megadepth_depths'
        else:
            raise NotImplementedError

        self.background_filenames = background_filenames
        self.bg_dataset_path = bg_dataset_path
        self.disparity_from_MiDas = disparity_from_MiDas

    def load_images(self, idx, do_flip=False):
        """ Load an image to use as left and a random background image to fill in occlusion holes"""

        filename = self.filenames[idx]
        image = self.loader(filename)
        print(f"original image shape : {image.width}, {image.height}")
        # image = image.resize((int(image.width // 2), (int(image.height // 2))), resample=pil.Resampling.NEAREST)
        print(f"current image shape: {image.width}, {image.height}")
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # frame = random.choice(self.filenames)
        # background = self.loader(frame)

        folder, frame = random.choice(self.background_filenames).split()
        background = self.loader(os.path.join(self.bg_dataset_path, folder, frame + '.jpg'))

        return image, background, filename

    def load_disparity(self, idx, do_flip=False):
        frame = self.filenames[idx].split('/')[-1].split('.')[0]
        disparity = None
        try:
            # disparity = self.read_pfm(os.path.join(self.disparity_from_MiDas, frame + '.pfm'))
            # disparity_path = os.path.join(self.disparity_from_MiDas, f'{frame}.npy')
            disparity_path = os.path.join(self.disparity_from_MiDas, f'{frame}_depth.png')
            print(disparity_path )
            # disparity = np.load(disparity_path)
            disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            disparity[np.isinf(disparity)] = 0.0
            disparity[np.isnan(disparity)] = 0.
            print(f"original max depth: {disparity.max()}")
            print(f"original min depth: {disparity.min()}")
            # disparity *= 192 / 80
            disparity *= 192 / 255
            print(f"current max depth: {disparity.max()}")
            print(f"current min depth: {disparity.min()}")
            print("disparity shape:", *disparity.shape)
        except:
            print(frame)
            raise EOFError
        # disparity = cv2.resize(disparity, dsize=None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
        # loaded disparity contains infs for no reading
        disparity[disparity == np.inf] = 0

        if do_flip:
            disparity = disparity[:, ::-1]
        return np.ascontiguousarray(disparity)

def save_img(savename, img):
    # im = pil.fromarray(img)
    im = pil.fromarray(np.uint8(img))
    im.save(savename)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from build_data_local import *
    dataset_root = '/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/'
    split = "samples"
    sensor = "CAM_FRONT"
    dataset_path = os.path.join(dataset_root, "nuscenes_fqj", split, sensor)
    disparity_from_MiDas = os.path.join(dataset_root, "output_4", split, sensor)

    # 过滤已经生成的图片
    #target_root = os.path.join(dataset_root, "output_4", split, sensor)
    target_root = os.path.join(dataset_root, "output_anything", split, sensor)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    target_names = set()
    for name in os.listdir(target_root):
        _name = '_'.join(name.split('_')[:2])
        target_names.add(_name)
    #target_names.add('2018-10-23-13-59-11_2018-10-23-14-46-37-394')
    #target_names = set()
    disparity_filenames = set()
    for filename in os.listdir(disparity_from_MiDas):
        disparity_filenames.add(filename.split('.')[0].replace('_left', '').replace('_right', '').replace('_disp', ''))
    print(disparity_filenames)
    disparity_from_MiDas = os.path.join(dataset_root, "nuscens_dpthanything_gt", split, sensor)
    # print(disparity_filenames)

    train_filenames = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) \
             if filename.split('.')[0] not in target_names and filename.split('.')[0] in  disparity_filenames]
    # train_filenames = train_filenames[10:20]
    # train_filenames = ['/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes_fqj/samples/CAM_FRONT/n015-2018-07-18-11-18-34+0800__CAM_FRONT__1531884525912460.jpg']
    # 重新生成一些不完整出错的图片
    # train_filenames = []
    # with open('mono_invalid.txt', 'r') as fl:
    #     for line in fl.readlines():
    #         _line = line.strip('\n').split('/')[-1].replace('_left.png', '.jpg')
    #         _line = os.path.join(dataset_path, _line)
    #         assert os.path.isfile(_line)
    #         train_filenames.append(_line)
    # print(train_filenames)。

    # train_filenames = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    print('len train_filename', len(train_filenames), train_filenames[0])

    # background file
    dataset_type = 'mscoco'
    bg_dataset_path = '/mnt/nas/public_data/coco'
    background_filenames = readlines(os.path.join('/mnt/nas/algorithm/xianda.guo/code/stereo-from-mono', \
                                                  'splits', dataset_type, 'train_files_all.txt'))

    # 保存生成的双目数据 左视图，右视图和视差
    save_root = target_root
    Drivingstereo = DrivingstereoDataset(dataset_path, bg_dataset_path, train_filenames, \
                                         background_filenames, disparity_from_MiDas, \
                                         feed_height=900, feed_width=1600,
                                         disable_sharpening=True, max_disparity=192, \
                                         disable_normalisation=True)

    _Drivingstereo = DataLoader(Drivingstereo, num_workers=32, batch_size=32)
    build_stero(Drivingstereo, save_root, flg_names=True)


