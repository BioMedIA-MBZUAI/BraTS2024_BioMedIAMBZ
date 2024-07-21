import os
import nibabel as nib
import numpy as np
import torch
import shutil

from pathlib import Path

from tqdm import tqdm

from monai import transforms

from biomedmbz_glioma.postprocessing import *

class Args:
    path_zip = "val-phase-predictions.zip" # output filename/path (this can be submitted to the validation leaderboard)
    dir_orig_data = '/share/sda/adam/ISBI2024-BraTS-GoAT/ISBI2024-BraTS-GoAT-ValidationData' # dir of original data
    dir_data = '/share/sda/adam/ISBI2024-BraTS-GoAT/val' # dir of preprocessed data
    dir_pred = 'tmp_pred/ft_mednext_m5' # validation-phase-get-predictions output
    dir_post = "tmp_post" # a temporary folder

args = Args()
torch.multiprocessing.set_sharing_strategy('file_system')

fn_postprocessing = transforms.Compose([
    AdvancedAsDiscrete(tc_threshold=0.5, wt_threshold=0.5, et_threshold=0.5),
    AdvanceETPost(2, 100, 0.1, 10, 'size', 26),
    AdvanceTCPost(0, 2, 150, 0.1, 10, 'size', 26),
    AdvanceWTPost(1, 0, 2, 500, 0.1, 10, 'size', 26),
    lambda x: torch.from_numpy(x['pred']),
])

if __name__ == '__main__':
    if os.path.exists(args.dir_post) and os.path.isdir(args.dir_post):
        shutil.rmtree(args.dir_post)
    os.makedirs(args.dir_post)
    
    for filename in tqdm(os.listdir(args.dir_pred)):
        name = filename.split('.')[0]
        
        pred = np.load(os.path.join(args.dir_pred, filename))
        mri  = np.load(os.path.join(args.dir_data, f'{name}_x.npy'))
        meta = np.load(os.path.join(args.dir_data, f'{name}_meta.npy'))
        
        seg = fn_postprocessing({'prob': pred, 'mri': mri, 'filename': filename.split('.')[0]})
        
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 3
        
        min_d, max_d = meta[0, 0], meta[1, 0]
        min_h, max_h = meta[0, 1], meta[1, 1]
        min_w, max_w = meta[0, 2], meta[1, 2]
        n_class, original_shape, cropped_shape = seg_out.shape[0], meta[2], meta[3]
        
        if not all(cropped_shape == seg_out.shape):
            raise ValueError
        
        final_seg = np.zeros(original_shape)
        final_seg[min_d:max_d, min_h:max_h, min_w:max_w] = seg_out
        final_seg = final_seg.astype(np.uint8)
        
        img = nib.load(os.path.join(args.dir_orig_data, name, f'{name}-t2f.nii.gz'))
        
        nib.save(
            nib.Nifti1Image(final_seg, img.affine, header=img.header),
            os.path.join(args.dir_post, f'{name}.nii.gz'),
        )
    
    if os.path.exists(args.path_zip):
        os.remove(args.path_zip)
    shutil.make_archive(args.path_zip.split('.')[0], 'zip', root_dir=args.dir_post)