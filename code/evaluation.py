from libs.logger import init_logger
from libs.config import config
from libs.utils import get_device
from libs.audio import WaveReader
from module import BSRNN
import torch
from torch import nn
from typing import Union
import os
from loguru import logger
import time
import numpy as np
import argparse
from pesq import pesq
from libs.sepm import composite as composite_py
import glob
import soundfile as sf
from torch.utils.data import Dataset

SAMPLING_RATE = 16000
def get_epoch(cp) -> int:
    return int(os.path.basename(cp).split(".")[0].split("_")[-1])

def parse_epoch_type(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        assert value in ("best", "latest")
        return value

class My_dataset(Dataset):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp, ref_scp, sample_rate):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
            
    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]
        return {
            "mix": mix.astype(np.float32),
            "ref": ref.astype(np.float32),
			"key": key
        }
    def __len__(self):
        return len(self.mix)

def main(args):
    log_file = os.path.join(args.model_base_dir, args.log_file)
    init_logger(file=log_file, level=args.log_level, model=args.model_base_dir)
    config.load(os.path.join(args.model_base_dir, "config.ini"))
    n_fft = config("n_fft", 512, int, section="data")
    n_hop = config("n_hop", 128, int, section="data")
    dev = get_device()
    model = BSRNN(num_channel=64, num_layer=5)
    
    checkpoint = os.path.join(args.model_base_dir, 'checkpoints')
    checkpoints = glob.glob(os.path.join(checkpoint, f"model_bsrnn*.ckpt.{args.epoch}"))
    model_path = max(checkpoints, key=get_epoch)
    epoch = get_epoch(model_path)
    logger.info("Found checkpoint {} with epoch {}".format(model_path, epoch))
    logger.info("Running on device {}".format(dev))
    model.load_state_dict((torch.load(model_path, map_location="cpu")))
    logger.info("Model loaded")
    if args.save_track and args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    total_cnt = 0
    total_PESQ = total_SISNR = total_CSIG = total_CBAK = total_COVL = 0
    total_PESQi = total_SISNRi = total_CSIGi = total_CBAKi = total_COVLi = 0
    ds = My_dataset(args.mix_scp, args.ref_scp, sample_rate=SAMPLING_RATE)
    start = time.time()
    for i, data_dict in enumerate(ds):
        name = data_dict['key']
        mix_16 = data_dict["mix"]
        ref_16 = data_dict["ref"]
        est_16 = enhance(model, dev, mix_16, name, n_fft, n_hop, args.save_track, args.save_dir)
        if args.save_track:
            continue      
        PESQ, PESQi = cal_PESQi(est_16, ref_16, mix_16)
        SISNR, SISNRi = cal_SISDRi(est_16, ref_16, mix_16)
        CSIG, CBAK, COVL, CSIGi, CBAKi, COVLi= cal_compositei(est_16, ref_16, mix_16)
        logger.info("Utt={:d} | fname={} | PESQ={:.3f} | PESQi={:.3f} | " 
                    "SISNR={:.3f} | SISNRi={:.3f} | CSIG={:.3f} | CSIGi={:.3f} |" 
                    "CBAK={:.3f} | CBAKi={:.3f} | COVL={:.3f} | COVLi={:.3f} | "
                    .format(total_cnt+1, name, PESQ, PESQi, SISNR, SISNRi, CSIG,\
                         CSIGi, CBAK, CBAKi, COVL, COVLi))  
        total_PESQ += PESQ
        total_PESQi += PESQi
        total_SISNR += SISNR
        total_SISNRi += SISNRi
        total_CSIG += CSIG
        total_CSIGi += CSIGi
        total_CBAK += CBAK
        total_CBAKi += CBAKi
        total_COVL += COVL
        total_COVLi += COVLi
        total_cnt += 1  
    end = time.time()
    if total_cnt != 0:
        logger.info("Average PESQ: {:.3f}".format(total_PESQ / total_cnt))
        logger.info("Average PESQi: {:.3f}".format(total_PESQi / total_cnt))
        logger.info("Average SISNR: {:.3f}".format(total_SISNR / total_cnt))
        logger.info("Average SISNRi: {:.3f}".format(total_SISNRi / total_cnt))
        logger.info("Average CSIG: {:.2f}".format(total_CSIG / total_cnt))
        logger.info("Average CSIGi: {:.2f}".format(total_CSIGi / total_cnt))
        logger.info("Average CBAK: {:.2f}".format(total_CBAK / total_cnt))
        logger.info("Average CBAKi: {:.2f}".format(total_CBAKi / total_cnt))
        logger.info("Average COVL: {:.2f}".format(total_COVL / total_cnt))
        logger.info("Average COVLi: {:.2f}".format(total_COVLi / total_cnt))
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))

@torch.no_grad()
def enhance(model: nn.Module, dev: get_device, audio: np.ndarray, \
            name: str, n_fft=512, hop=128, save_track=False, save_dir=None):
    model.to(dev)
    model.eval()
    noisy_pad = np.pad(audio, hop, mode='reflect')
    noisy_pad = torch.Tensor(noisy_pad).unsqueeze(0).to(dev)
    length = len(audio)
    noisy_spec = torch.stft(noisy_pad, n_fft, hop, window=torch.hann_window(n_fft).to(dev), return_complex=True)
    noisy_spec = torch.view_as_real(noisy_spec)
    est_spec = torch.view_as_complex(model(noisy_spec).detach())
    est_audio = torch.istft(est_spec, n_fft, hop, window=torch.hann_window(n_fft).to(dev))
    est_audio = torch.flatten(est_audio[:,hop:length+hop]).cpu().numpy()
    assert len(est_audio) == length
    if save_track:
        norm = np.linalg.norm(audio, np.inf)
        est_audio = est_audio*norm/np.max(np.abs(est_audio))
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, name)
        sf.write(saved_path, est_audio, SAMPLING_RATE)
    return est_audio

def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode ='wb'
    p = pesq(SAMPLING_RATE, ref, est, mode)
    return p

def cal_PESQi(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    pesq1 = cal_PESQ(est, ref)
    pesq2 = cal_PESQ(mix, ref)
    return pesq1, pesq1 - pesq2

def cal_SISDR(estimate: np.ndarray, reference: np.ndarray):
    reference = reference.reshape(-1, 1)
    estimate = estimate.reshape(-1, 1)
    eps = np.finfo(reference.dtype).eps
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps + Sss) / (eps + Snn))
    return sisdr

def cal_SISDRi(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_SISDR(est, ref)
    stoi2 = cal_SISDR(mix, ref)
    return stoi1, stoi1 - stoi2

def cal_composite(est, ref, sr=SAMPLING_RATE):
    assert len(est) == len(ref)
    pesq_mos, Csig, Cbak, Covl, segSNR = composite_py(ref, est, sr)
    return pesq_mos, Csig, Cbak, Covl, segSNR

def cal_compositei(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    pesq_mos1, Csig1, Cbak1, Covl1, segSNR1 = cal_composite(est, ref)
    pesq_mos2, Csig2, Cbak2, Covl2, segSNR2 = cal_composite(mix, ref)
    return Csig1, Cbak1, Covl1, Csig1-Csig2, Cbak1-Cbak2, Covl1-Covl2


def setup_df_argument_parser(default_log_level: str = "INFO"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default="/Share/wsl/exp/BSRNN/exp0/base_dir",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=default_log_level,
    )
    parser.add_argument(
        "--epoch",
        "-e",
        default="best",
        type=parse_epoch_type,
    )
    parser.add_argument(
        "-s",
        "--save_track",
        action='store_true',
        # type=bool,
    )
    parser.add_argument(
        "--save_dir",
        default="/Share/wsl/data",
        type=str,
    )
    return parser

def run():
    parser = setup_df_argument_parser()
    parser.add_argument(
        "--mix_scp",
        type=str,
        default="/Share/wsl/data/VCTK_16k/path/demo/train_dir/noisy.scp",
    )
    parser.add_argument(
        "--ref_scp",
        type=str,
        default="/Share/wsl/data/VCTK_16k/path/demo/train_dir/clean.scp",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default='enhance.log',
    )
    args = parser.parse_args()
    main(args)
if __name__ == "__main__":
    run()
