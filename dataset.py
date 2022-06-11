import argparse
import sys
import os
import glob
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

def error(msg):
    print('Error: ' + msg)
    exit(1)

class ImageProcess:
    def __init__(self, process_dir, expected_images, print_progress=True, progress_interval=10):
        self.process_dir       = process_dir
        self.pyt_prefix         = os.path.join(self.process_dir, os.path.basename(self.process_dir))
        self.expected_images    = expected_images
        self.shape              = None
        self.resolution_log2    = None
        self.pyt_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % process_dir)
        if not os.path.isdir(self.process_dir):
            os.makedirs(self.process_dir)
        assert os.path.isdir(self.process_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for writer in self.pyt_writers:
            writer.close()
        self.pyt_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)

    #返回<class 'numpy.ndarray'>的0到expected_images的乱序矩阵
    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    #将1024的图片分别按分辨率4*4， 8*8，...，1024*1024存储
    def add_image(self, img):
        if self.shape is None:
            self.shape = img.shape  #[3, 1024, 1024]
            self.resolution_log2 = int(np.log2(self.shape[1]))  #10
            print('11111:', self.resolution_log2)
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
        assert img.shape == self.shape
        image_resolution_list = [[], [], [], [], [], [], [], [], []]
        for i in range(len(image_resolution_list)):
            if i:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            image_resolution_list[len(image_resolution_list) - i - 1].append(quant)
        return image_resolution_list



    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def create_from_images(image_process_dir, image_dir, shuffle, need_save=0):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))   #['FFHQ_image\\00000.png',..., 'FFHQ_image\\10000.png']
    if len(image_filenames) == 0:
        error('No input images found')

    # 以图片00000.png为例测试输入的图片是否是(1024, 1024, 3)
    img = np.asarray(PIL.Image.open(image_filenames[0]))  #处理00000.png  img.shape: (1024, 1024, 3)
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    image_out = []

    if need_save == 1:
        # 将1024的图片分别按分辨率4*4， 8*8，...，1024*1024存储
        image_resolution_list = [[], [], [], [], [], [], [], [], []]
        with ImageProcess(image_process_dir, len(image_filenames)) as imagepro:
            order = imagepro.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
            for idx in range(order.size):
                img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
                if channels == 1:
                    img = img[np.newaxis, :, :] # HW => CHW
                else:
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                for i in range(len(image_resolution_list)):
                    if i:
                        img = img.astype(np.float32)
                        img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
                    quant = np.rint(img).clip(0, 255).astype(np.uint8)
                    image_resolution_list[len(image_resolution_list) - i - 1].append(quant)
        for i in range(2, 11):
            filename = os.path.join(image_process_dir, 'resolution_%d' % i)
            np.save(filename, image_resolution_list[i-2])

    with ImageProcess(image_process_dir, len(image_filenames)) as imagepro:
        order = imagepro.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            img = img.transpose([2, 0, 1])
            image_out.append(img)
    return image_out


def execute_cmdline(argv):
    # argv = datasetool.py  create_from_images  FFHQ_data FFHQ_iamge
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.',
        epilog='Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)
    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)
    print(11111)
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

class ImageDataset(Dataset):
    def __init__(self, x):
        # super(ImageDataset, self).__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> T_co:
        return torch.tensor(self.x[idx]).float()

# 得到不同分辨率的图像
def get_resolution(img):
    image_resolution_list = [[], [], [], [], [], [], [], [], []]
    for i in range(len(image_resolution_list)):
        if i:
            img = (img[:, :, 0::2, 0::2] + img[:, :, 0::2, 1::2] + img[:, :, 1::2, 0::2] + img[:, :, 1::2, 1::2]) * 0.25
        image_resolution_list[len(image_resolution_list) - i - 1].append(img)
    return image_resolution_list

if __name__ == "__main__":
    _ = create_from_images('FFHQ_data', 'FFHQ', shuffle=1, need_save= 0)