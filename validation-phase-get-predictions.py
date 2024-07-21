import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import multiprocessing

from threading import Thread
from tqdm import tqdm

from biomedmbz_glioma.inference import InferenceModule
from biomedmbz_glioma.dataset import get_test_dataset

from monai.data import DataLoader

from model_directory import *

# weighting_mednext_m3   = [1.0, 0.0, 0.5] # weight for ensembling
weighting_mednext_m5   = [1.0, 1.0, 1.0] # weight for ensembling

class Args:
    dir_pred = '/share/sda/adam/ISBI2024-BraTS-GoAT/tmp_pred/mednext_m5' # dir to save output tumor probability maps
    list_models = [
        [get_model_mednext_m5, '/share/sda/adam/ISBI2024-BraTS-GoAT/weights/m-m_k-5_f-0/epoch=71-step=64800.ckpt', weighting_mednext_m5],
        [get_model_mednext_m5, '/share/sda/adam/ISBI2024-BraTS-GoAT/weights/m-m_k-5_f-1/epoch=82-step=74700.ckpt', weighting_mednext_m5],
        [get_model_mednext_m5, '/share/sda/adam/ISBI2024-BraTS-GoAT/weights/m-m_k-5_f-2/epoch=84-step=76500.ckpt', weighting_mednext_m5],
        [get_model_mednext_m5, '/share/sda/adam/ISBI2024-BraTS-GoAT/weights/m-m_k-5_f-3/epoch=91-step=82800.ckpt', weighting_mednext_m5],
        [get_model_mednext_m5, '/share/sda/adam/ISBI2024-BraTS-GoAT/weights/m-m_k-5_f-4/epoch=75-step=68400.ckpt', weighting_mednext_m5],
        # [get_model_mednext_m3, ..., ...], # for ensembling. You can ensemble models as many as you want. However, this code currently doesn't allow varying patch sizes
    ]
    data_dir = '/share/sda/adam/ISBI2024-BraTS-GoAT/val' # preprocessed data dir
    infer_overlap=0.5
    tta=True
    blend='gaussian'
    sw_batch_size=2
    num_workers=2
    roi_x=128
    roi_y=128
    roi_z=128
    cuda=True
    pin_memory=False
    seed=42

args = Args()
torch.multiprocessing.set_sharing_strategy('file_system')

roi_size = (args.roi_x, args.roi_y, args.roi_z)

test_dataset = get_test_dataset(args.data_dir)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=args.num_workers, drop_last=False,
    pin_memory=args.pin_memory,
)

def update_npy(path, pred, weighting):
    pred = pred.copy()
    
    for i in range(len(weighting)):
        pred[i] = pred[i] * weighting[i]
    
    if not os.path.exists(path):
        np.save(path, pred)
    else:
        cur_pred = np.load(path) + pred
        np.save(path, cur_pred)

def postprocess_npy(path):
    pred = np.load(path)
    
    weighting = np.array([weighting for _, _, weighting in args.list_models]).sum(axis=0)
    
    prob = pred.copy()
    for i in range(weighting.shape[0]):
        prob[i] = prob[i] / weighting[i]
    
    np.save(path, prob)

def predict(ModelClass, ckpt_path, weighting, save_dir):
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model  = ModelClass()
    module = InferenceModule(model, not model.apply_sigmoid, roi_size, args.infer_overlap, args.sw_batch_size, nn.Identity(), tta=args.tta, blend=args.blend)
    
    try:
        module.load_state_dict(checkpoint['state_dict'], strict=True)
    except:
        module.model.load_state_dict(checkpoint['state_dict'], strict=True)
    module.eval()
    
    if args.cuda: module.cuda()
    
    threads = []
    for sample in tqdm(test_dataloader):
        with torch.no_grad():
            preds = module(sample["image"].cuda()) if args.cuda else module(sample["image"])
            
            for name_2023, pred in zip(sample['name'], preds):
                thread = Thread(target=update_npy, args=(os.path.join(save_dir, f'{name_2023}.npy'), pred.numpy(), weighting,))
                thread.start()
                threads.append(thread)
    
    if args.cuda: module.cpu()
    
    for thread in threads:
        thread.join()

# if os.path.exists(args.dir_pred) and os.path.isdir(args.dir_pred):
#     shutil.rmtree(args.dir_pred)
os.makedirs(args.dir_pred)

for model_info in args.list_models:
    predict(model_info[0], model_info[1], model_info[2], args.dir_pred)

pool = multiprocessing.Pool(processes=8)
pool.starmap(postprocess_npy, [(os.path.join(args.dir_pred, name_2023),) for name_2023 in os.listdir(args.dir_pred)],)
pool.close()

# scp -r ft_m-m_k-5_s-14 fadillahmaani@10.127.30.108:/share/sda/adam/ISBI2024-BraTS-GoAT/weights/ft_m-m_k-5_s-14