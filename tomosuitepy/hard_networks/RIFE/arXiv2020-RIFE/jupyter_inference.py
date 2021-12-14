import os, sys, warnings, cv2, argparse, shutil
import tifffile as tif
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
_ = os.path.dirname(__file__).split('/')[:-3]
sys.path.append(f"{'/'.join(_)}/")
from base.common import load_extracted_prj

class Dummy():
    pass

def jupyter_rife(basedir, exp=2, output='frames', gpu='0', sparse=1):
    
    args = Dummy()
    args.basedir = basedir
    args.exp = exp
    args.output = output
    args.gpu = gpu
    args.sparse = sparse

    args.ratio = float(0.0) # inference ratio between two images with 0 - 1 range
    args.rthreshold = float(0.02) # returns image when actual ratio falls in given range threshold
    args.rmaxcycles = int(8) # limit max number of bisectional cycles

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    import torch
    from torch.nn import functional as F
    from model.RIFE_HDv2 import Model

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = Model()
    model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log'), -1)
    model.eval()
    model.device()

    # Load in all the images
    total_prjs = load_extracted_prj(args.basedir)
    total_prjs = total_prjs[::args.sparse]
    save_location = f"{args.basedir}rife/{args.output}/"
    if os.path.isdir(save_location):
        shutil.rmtree(save_location)
    os.mkdir(save_location)
    prj_max = total_prjs.max()
    total_prjs = total_prjs / prj_max
    total_prjs = total_prjs * 255.0

    # Iterate over the images
    zfill_val = len(str(len(total_prjs) * 2**args.exp))
    current_frame = 0

    for iteration in tqdm(range(0, len(total_prjs)-1), desc='Interpolation'):
        # Tripple the image arrays
        img0 = np.dstack((total_prjs[iteration], total_prjs[iteration], total_prjs[iteration]))
        img1 = np.dstack((total_prjs[iteration + 1], total_prjs[iteration + 1], total_prjs[iteration + 1]))

        img0 = img0.astype(np.float32)
        img1 = img1.astype(np.float32)

        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if args.ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if args.ratio <= img0_ratio + args.rthreshold / 2:
                middle = img0
            elif args.ratio >= img1_ratio - args.rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(args.rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                        break
                    if args.ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        if not os.path.exists(save_location):
            os.mkdir(save_location)
        for i in range(len(img_list)):
            im2save = (img_list[i][0] * prj_max).cpu().numpy().transpose(1, 2, 0)[:h, :w]
            im2save = rgb2gray(im2save)
            tif.imsave(f'{save_location}/img_{str(current_frame).zfill(zfill_val)}.tif', im2save)
            current_frame += 1