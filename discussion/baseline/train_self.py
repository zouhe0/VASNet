# =====================================================================
# train_self.py — Two-phase Self-Supervised Pansharpening
# =====================================================================
# --mode full: full-size training (default)
# --mode lr:   reduced-resolution training
# =====================================================================

import argparse, time, os, sys, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.backends.cudnn as cudnn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import scipy.io as sio, h5py

from data import Dataset
from mymodel import SDNetFusionNet_All, SDNetFusionNet_Conv
from loss import LossCalculator
import warnings
warnings.filterwarnings("ignore")

SEED = 10
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--lr_phase1", type=float, default=0.015)
parser.add_argument("--epochs_phase1", type=int, default=240)
parser.add_argument("--lr_phase2", type=float, default=0.01)
parser.add_argument("--epochs_phase2", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--w_spa", type=int, default=200)
parser.add_argument("--w_spec", type=int, default=250)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_id", type=int, default=None)
parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=19)
parser.add_argument("--sensor", type=str, default=None)
parser.add_argument("--ratio", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--lmbd_weight", type=float, default=500.0)
parser.add_argument("--dataset", type=str, default="WV3", choices=["WV3","WV2","QB","GF2"])
parser.add_argument("--mode", type=str, default="full", choices=["lr","full","both"])
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--ablation", type=str, default="none",
    choices=["none","no_spatial","no_spectral","no_phase1","no_csc","fix_lmbd"])
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
ratio = args.ratio
DATASET_PRESETS = {
    "WV3": {"sensor":"WV3","path":"/media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5"},
    "WV2": {"sensor":"WV2","path":"/media/zouhe/Elements/Data/PanCollection/test_data/test_wv2_OrigScale_multiExm1.h5"},
    "QB": {"sensor":"QB","path":"/media/zouhe/Elements/Data/PanCollection/test_data/test_qb_OrigScale_multiExm1.h5"},
    "GF2": {"sensor":"WV4","path":"/media/zouhe/Elements/Data/PanCollection/test_data/test_gf2_OrigScale_multiExm1.h5"},
}
dataset_name = args.dataset.upper(); preset = DATASET_PRESETS[dataset_name]
sensor = args.sensor.upper() if args.sensor else preset["sensor"]
data_path = args.data_path if args.data_path else preset["path"]
if args.mode == "lr": data_path = data_path.replace("_OrigScale","")
with h5py.File(data_path,"r") as f: spectral_num = f["ms"].shape[1]

w_var = 1850000.0; w_spa = args.w_spa; w_spec = args.w_spec
save_dir_base = "result_self"
os.makedirs(save_dir_base, exist_ok=True)
if args.ablation != "none": save_dir_base = os.path.join("result_abla",args.ablation); os.makedirs(save_dir_base,exist_ok=True)

def get_save_dir(mode):
    d = os.path.join(save_dir_base, f"{dataset_name}_{mode}"); os.makedirs(d,exist_ok=True); return d
os.makedirs("model_FUG",exist_ok=True)

def get_all_lmbd_values(model):
    vals = []
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock': vals.append(m.lmbd.item())
    return vals

def compute_lmbd_regularization(model):
    total = 0.0
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock': total = total + m.lmbd.pow(2).sum()
    return total

def freeze_lmbd_to_01(model):
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            m.lmbd = nn.Parameter(torch.tensor([0.1],device=m.lmbd.device),requires_grad=False)

def clamp_lmbd_nonneg(model):
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock': m.lmbd.data.clamp_(min=0.0,max=10)

def norm_pan_to_4d(pan_tensor):
    if pan_tensor.dim()==2: return pan_tensor.unsqueeze(0).unsqueeze(0)
    elif pan_tensor.dim()==3: return pan_tensor.unsqueeze(0) if pan_tensor.shape[0]==1 else pan_tensor.unsqueeze(1)
    elif pan_tensor.dim()==4: return pan_tensor
    raise ValueError("bad pan dim")

def hwc_save(tensor_4d, scale=2047.0):
    return tensor_4d.squeeze(0).permute(1,2,0).cpu().numpy()*scale

def phase1_pretrain(data_id, mode):
    save_dir = get_save_dir(mode)
    file_prefix = f"{data_id}_self"
    model_prefix = f"{dataset_name}_{data_id}_self_{mode}"
    with h5py.File(data_path,"r") as f:
        ms_np = np.array(f["ms"][data_id],dtype=np.float32)/2047.0
        lms_np = np.array(f["lms"][data_id],dtype=np.float32)/2047.0
        pan_np = np.array(f["pan"][data_id],dtype=np.float32)/2047.0
        gt_np = np.array(f["gt"][data_id],dtype=np.float32) if "gt" in f else None
    ms = torch.from_numpy(ms_np).to(device).clamp(0,1)
    lms = torch.from_numpy(lms_np).to(device).clamp(0,1)
    pan_raw = torch.from_numpy(pan_np).to(device).clamp(0,1)
    while pan_raw.dim()>2: pan_raw = pan_raw.squeeze(0)
    if pan_raw.dim()==1: side=int(np.sqrt(pan_raw.shape[0])); pan_raw=pan_raw.reshape(side,side)
    pan_4d = norm_pan_to_4d(pan_raw); lms_4d = lms.unsqueeze(0); ms_4d = ms.unsqueeze(0)
    pan_lr = F.interpolate(pan_4d,size=ms.shape[1:],mode="bilinear",align_corners=False)
    ms_low = F.interpolate(ms_4d,scale_factor=1.0/ratio,mode="bilinear",align_corners=False)
    lms_lr = F.interpolate(ms_low,size=ms.shape[1:],mode="bilinear",align_corners=False)
    model = (SDNetFusionNet_Conv(spectral_num=spectral_num) if args.ablation=="no_csc" else SDNetFusionNet_All(spectral_num=spectral_num)).to(device)
    if args.ablation=="fix_lmbd": freeze_lmbd_to_01(model)
    opt = optim.Adam(model.parameters(),lr=args.lr_phase1,betas=(0.9,0.999))
    sched = CosineAnnealingLR(opt,T_max=args.epochs_phase1,eta_min=args.lr_phase1*0.01)
    min_loss = float("inf"); best_path = os.path.join("model_FUG",f"{model_prefix}_phase1_best.pth")
    for epoch in range(1,args.epochs_phase1+1):
        model.train(); opt.zero_grad()
        res = model(lms_lr,pan_lr); output = res + lms_lr
        loss = torch.mean((output-ms_4d)**2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        opt.step(); sched.step()
        if args.ablation not in ("no_csc","fix_lmbd"): clamp_lmbd_nonneg(model)
        if loss.item()<min_loss: min_loss=loss.item(); torch.save(model.state_dict(),best_path)
    model.load_state_dict(torch.load(best_path,map_location=device)); model.eval()
    with torch.no_grad(): p1_out_lr = (model(lms_lr,pan_lr)+lms_lr).clamp(0,1)
    p1_lr_dict = {"I_MS_LR":ms.permute(1,2,0).cpu().numpy()*2047.0,"I_MS":lms.permute(1,2,0).cpu().numpy()*2047.0,"I_PAN":pan_raw.cpu().numpy()*2047.0,"proposed":hwc_save(p1_out_lr)}
    if gt_np is not None: p1_lr_dict["gt"]=gt_np.transpose(1,2,0)
    sio.savemat(os.path.join(save_dir,f"{file_prefix}_phase1_lr.mat"),p1_lr_dict)
    del model
    model_full = (SDNetFusionNet_Conv(spectral_num=spectral_num) if args.ablation=="no_csc" else SDNetFusionNet_All(spectral_num=spectral_num)).to(device)
    if args.ablation=="fix_lmbd": freeze_lmbd_to_01(model_full)
    model_full.load_state_dict(torch.load(best_path,map_location=device)); model_full.eval()
    with torch.no_grad(): teacher_out = (model_full(lms_4d,pan_4d)+lms_4d).clamp(0,1)
    teacher_dict = {"sr":hwc_save(teacher_out)}
    if gt_np is not None: teacher_dict["gt"]=gt_np.transpose(1,2,0)
    sio.savemat(os.path.join(save_dir,f"{file_prefix}_teacher.mat"),teacher_dict)
    p1_full_dict = {"I_MS_LR":ms.permute(1,2,0).cpu().numpy()*2047.0,"I_MS":lms.permute(1,2,0).cpu().numpy()*2047.0,"I_PAN":pan_raw.cpu().numpy()*2047.0,"proposed":hwc_save(teacher_out)}
    if gt_np is not None: p1_full_dict["gt"]=gt_np.transpose(1,2,0)
    sio.savemat(os.path.join(save_dir,f"{file_prefix}_phase1_full.mat"),p1_full_dict)
    teacher_tensor = teacher_out.squeeze(0).detach()
    gt_hw = gt_np.transpose(1,2,0) if gt_np is not None else None
    del model_full
    return teacher_tensor, lms, pan_raw, ms, gt_hw

def phase2_distill(data_id, teacher_tensor, lms, pan_raw, ms, mode, gt_hw=None):
    save_dir = get_save_dir(mode)
    file_prefix = f"{data_id}_self"
    model_prefix = f"{dataset_name}_{data_id}_self_{mode}"
    train_set = Dataset(data_path,data_id)
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    model = (SDNetFusionNet_Conv(spectral_num=spectral_num) if args.ablation=="no_csc" else SDNetFusionNet_All(spectral_num=spectral_num)).to(device)
    if args.ablation=="fix_lmbd": freeze_lmbd_to_01(model)
    loss_calculator = LossCalculator(sensor=sensor,ratio=ratio,N=41,device=device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr_phase2,betas=(0.9,0.999))
    sched_phase2 = CosineAnnealingLR(optimizer,T_max=args.epochs_phase2,eta_min=args.lr_phase2*0.01)
    min_total_loss=float("inf"); best_path=os.path.join("model_FUG",f"{model_prefix}_best.pth")
    for epoch in range(1,args.epochs_phase2+1):
        model.train(); epoch_loss_var,epoch_loss_spa,epoch_loss_spec=[],[],[]
        for i,batch in enumerate(train_loader):
            ms_b,lms_b,pan_b=batch[0].to(device),batch[1].to(device),batch[2].to(device)
            optimizer.zero_grad()
            if len(pan_b.shape)==3: pan_b=pan_b.unsqueeze(1)
            res_student=model(lms_b,pan_b); fusion_out=res_student+lms_b; fusion_out=fusion_out.squeeze(0)
            loss_var=torch.mean((fusion_out-teacher_tensor)**2)
            fusion_out_hw_c=fusion_out.permute(1,2,0); _,H,_=ms_b[0].shape; block_size=H//ratio
            loss_spa=loss_calculator.compute_spatial_fidelity_loss(fusion_out_hw_c,ms_b[0].permute(1,2,0),pan_b[0].squeeze(0),block_size,use_ergas=True)
            loss_spec=loss_calculator.compute_spectral_loss(fusion_out_hw_c,ms_b[0].permute(1,2,0))
            lmbd_reg=0.0 if args.ablation in ("no_csc","fix_lmbd") else compute_lmbd_regularization(model)
            if args.ablation=="no_spatial": total_loss=w_var*loss_var+w_spec*loss_spec-args.lmbd_weight*lmbd_reg
            elif args.ablation=="no_spectral": total_loss=w_var*loss_var+w_spa*loss_spa-args.lmbd_weight*lmbd_reg
            elif args.ablation=="no_phase1": total_loss=w_spa*loss_spa+w_spec*loss_spec-args.lmbd_weight*lmbd_reg
            else: total_loss=w_var*loss_var+w_spa*loss_spa+w_spec*loss_spec-args.lmbd_weight*lmbd_reg
            epoch_loss_var.append((loss_var*w_var).item()); epoch_loss_spa.append((loss_spa*w_spa).item()); epoch_loss_spec.append((loss_spec*w_spec).item())
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step(); sched_phase2.step()
            if args.ablation not in ("no_csc","fix_lmbd"): clamp_lmbd_nonneg(model)
        avg_total_loss=np.mean(epoch_loss_var)+np.mean(epoch_loss_spa)+np.mean(epoch_loss_spec)
        if avg_total_loss<min_total_loss: min_total_loss=avg_total_loss; torch.save(model.state_dict(),best_path)
    model.load_state_dict(torch.load(best_path,map_location=device)); model.eval()
    with torch.no_grad():
        lms_in=lms.unsqueeze(0); pan_in=norm_pan_to_4d(pan_raw)
        final_out=(model(lms_in,pan_in)+lms_in).clamp(0,1)
    result_dict={"I_MS_LR":ms.permute(1,2,0).cpu().numpy()*2047.0,"I_MS":lms.permute(1,2,0).cpu().numpy()*2047.0,"I_PAN":pan_raw.cpu().numpy()*2047.0,"proposed":hwc_save(final_out)}
    if gt_hw is not None: result_dict["gt"]=gt_hw
    sio.savemat(os.path.join(save_dir,f"{file_prefix}_result.mat"),result_dict)
    del model

def process_single_image(data_id):
    mode=args.mode
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    import random; random.seed(SEED)
    teacher_tensor,lms,pan_raw,ms,gt_hw=phase1_pretrain(data_id,mode)
    phase2_distill(data_id,teacher_tensor,lms,pan_raw,ms,mode,gt_hw)
    return True

def main():
    if args.data_id is None:
        for did in range(args.start_id,args.end_id+1):
            process_single_image(did)
    else:
        process_single_image(args.data_id)

if __name__=="__main__":
    main()
