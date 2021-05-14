import os
import argparse
import mmcv
import random
import numpy as np


def parse_args():
  parser = argparse.ArgumentParser(
      description='Create list dataset txt'
  )
  parser.add_argument('cityscapes_path', help='cityscapes data path')  
  parser.add_argument('--img-dir', default='leftImg8bit', type=str)
  parser.add_argument('--gt-dir', default='gtFine', type=str)
  parser.add_argument('-o','--out-dir', help='output path')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  cityscapes_path = args.cityscapes_path
  out_dir = args.out_dir if args.out_dir else cityscapes_path
  mmcv.mkdir_or_exist(out_dir)
  
  img_dir = os.path.join(cityscapes_path, args.img_dir)
  gt_dir = os.path.join(cityscapes_path, args.gt_dir)
  
  split_names = ['train','val', 'test']
  
  for split in split_names:

    imgs, gts = [], [] 
    file_names = []

    for img in mmcv.scandir(os.path.join(img_dir, split), '_leftImg8bit.png', recursive=True):
      img_pth = args.img_dir + '/'+ split +'/' + img
      imgs.append(img_pth)

    for gt in mmcv.scandir(os.path.join(gt_dir,split), '_labelIds.png',recursive=True):
      gt_pth = args.gt_dir + '/' + split + '/' + gt
      gts.append(gt_pth)

    imgs = np.sort(imgs)
    gts = np.sort(gts)

    for img_pth, gt_pth in zip(imgs, gts):
      file_names.append(','.join([img_pth, gt_pth]))
    
    random.shuffle(file_names)
    print(len(file_names))

    with open(os.path.join(out_dir, f'{split}.txt'),'w') as f:
      f.writelines(f + '\n' for f in file_names)
        
      
    
if __name__ == '__main__':
    main()
