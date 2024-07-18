import argparse
import torch
import random
import numpy as np
import os 
from loguru import logger
from module import * 
import glob
from collections import defaultdict
import librosa
import copy
from rVADfast import rVADfast
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from joblib import Parallel, delayed

from libs.logger import init_logger, log_metrics, log_model_summary
from libs.config import config
from libs.dataset import make_dataloader
from libs.loss import Loss
import onnxruntime as ort

EPS = np.finfo(float).eps
clipping_threshold=0.99
SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
vad = rVADfast()  

def same_seeds(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate=16000, is_mean=True):
        fs = sampling_rate
        len_samples = int(INPUT_LENGTH*fs)
        audio_repeat = audio
        while audio_repeat.shape[-1] < len_samples:
            audio_repeat = np.concatenate((audio_repeat, audio), axis=1)
        
        num_hops = int(np.floor(audio_repeat.shape[-1]/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_p808_mos = []
        if is_mean:
            for idx in range(num_hops):
                audio_seg = audio_repeat[:,int(idx*hop_len_samples):int((idx+INPUT_LENGTH)*hop_len_samples)]
                if audio_seg.shape[-1] < len_samples:
                    continue
                p808_input_features = np.transpose(self.audio_melspec(audio=audio_seg[:, :-160]).astype('float32'), (2, 0, 1))
                p808_oi = {'input_1': p808_input_features}
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0].mean()
                predicted_p808_mos.append(p808_mos)
            clip_dict = {}
            clip_dict['p808_mos'] = np.mean(predicted_p808_mos)
            return clip_dict
        else:
            for idx in range(num_hops):
                audio_seg = audio_repeat[:,int(idx*hop_len_samples):int((idx+INPUT_LENGTH)*hop_len_samples)]
                if audio_seg.shape[-1] < len_samples:
                    continue
                p808_input_features = np.transpose(self.audio_melspec(audio=audio_seg[:, :-160]).astype('float32'), (2, 0, 1))
                p808_oi = {'input_1': p808_input_features}
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0]
                predicted_p808_mos.append(p808_mos)
            predicted_p808_mos_averg = np.mean(predicted_p808_mos, axis=0)
            return predicted_p808_mos_averg.flatten()

def compute_vad(audio): 
    global vad
    vad_labels, vad_timestamps = vad(audio, SAMPLING_RATE)
    ratio_of_ones = np.mean(vad_labels == 1)
    return ratio_of_ones

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default='/Share/wsl/exp/BSRNN/exp0/base_dir', type=str)
    parser.add_argument("--log_level", default='INFO', type=str)
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_teacher_dir = os.path.join(args.base_dir, "checkpoints_teacher")
    log_level = args.log_level
    init_logger(file=os.path.join(args.base_dir, "train.log"), level = log_level, model=args.base_dir)
    config_file = os.path.join(args.base_dir, "config.ini")
    config.load(config_file)
    gpuid = config("gpuid", 0, str, section="train")
    gpuid = tuple(map(int, gpuid.split(",")))
    if not isinstance(gpuid, tuple):
        gpuid = (gpuid, )
    device = torch.device('cuda:{}'.format(gpuid[0]))
    logger.info("Running on device gpuid: {}".format(gpuid))
    max_epoch = config("max_epoch", 100, int, section="train") 
    bs:int = config("BATCH_SIZE", 120, int, section="train")
    bs_eval:int = config("BATCH_SIZE_EVAL", 100, int, section="train")
    num_workers = config("NUM_WORKERS", 32, int, section="train")
    log_freq = config("LOG_FREQ", 100, int, section="train")
    seed = config("SEED", 1, int, section="train")
    primary_model_path = config("primary_model_path", section="train")
    p808_model_path = config("p808_model_path", section="train")

    train_mix_dir = config("TRAIN_MIX_DIR", section="data")
    valid_mix_dir = config("valid_mix_dir", section="data")
    sample_rate = config("sample_rate", 16000, int, section="data")
    n_fft = config("n_fft", cast=int, section="data")
    n_hop = config("n_hop", cast=int, section="data")
    max_sample_len_s = config("max_sample_len_s", cast=int, section="data")

    init_lr = config("init_lr", cast=float, section="optim")
    patience = config("patience", cast=int, section="optim")
    factor = config("factor", cast=float, section="optim")

    chunk_size = max_sample_len_s * sample_rate
    same_seeds(seed)
    train_loader = make_dataloader(True,
                                    train_mix_dir,
                                    sample_rate,
                                    batch_size=bs,
                                    num_workers=num_workers,
                                    chunk_size=chunk_size,
                                    )
    valid_loader = make_dataloader(False,
                                    valid_mix_dir,
                                    sample_rate,
                                    batch_size=bs_eval,
                                    num_workers=num_workers,
                                    chunk_size=chunk_size,
                                    )
    teacher_model = BSRNN(num_channel=64, num_layer=5)
    student_model = BSRNN(num_channel=64, num_layer=5)

    teacher_model, _ = read_cp(teacher_model, checkpoint_teacher_dir, extension="ckpt", is_teacher=True)
    
    try:
        log_model_summary(student_model, device, n_fft, n_hop)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")

    trainer = Trainer(teacher_model, student_model, train_loader, valid_loader, device, gpuid, \
                      n_fft, n_hop, bs, log_freq, init_lr, \
                        checkpoint_dir, patience, factor, primary_model_path, p808_model_path
                        )
    trainer.train(0, max_epoch)


class Trainer:
    def __init__(self, teacher_model, student_model, train_dl, valid_dl, dev, gpuid, \
                 n_fft, n_hop, train_bs, log_freq, init_lr, \
                 checkpoint_dir, patience, factor, primary_model_path, p808_model_path
                    ):
        self.n_fft = n_fft
        self.hop = n_hop
        self.train_ds = train_dl
        self.valid_ds = valid_dl
        self.dev = dev
        self.train_bs = train_bs
        self.log_freq = log_freq
        self.gpuid = gpuid
        self.teacher_model = copy.deepcopy(teacher_model).to(self.dev)
        self.student_model = copy.deepcopy(student_model).to(self.dev)
        self.compute_score = ComputeScore(primary_model_path, p808_model_path)
        self.loss = Loss().to(self.dev)
        opt_model_paths = glob.glob(os.path.join(checkpoint_dir, "opt_bsrnn*"))
        if len(opt_model_paths) == 0:
            self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=init_lr)
        else:
            latest_opt_model_path = max(opt_model_paths, key=get_epoch)
            cpt_model = torch.load(latest_opt_model_path, map_location="cpu")
            self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=init_lr)
            self.optimizer.load_state_dict(cpt_model)
        self.scheduler = ReduceLROnPlateau(
                                self.optimizer,
                                mode="min",
                                factor=factor,
                                patience=patience,
                                min_lr=0,
                                verbose=True
                                )
        self.save_model_dir = checkpoint_dir
    
    def save_checkpoint(self, epoch, suffix=None, save_teacher=True):
        if suffix is not None:
            path = os.path.join(self.save_model_dir, 'model_bsrnn' + '_' + str(epoch) + '.ckpt.' + suffix)
        else:
            path = os.path.join(self.save_model_dir, 'model_bsrnn' + '_' + str(epoch) + '.ckpt')        
        opt_path =  os.path.join(self.save_model_dir, 'opt_bsrnn' + '_' + str(epoch) + '.ckpt')

        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if save_teacher:
            torch.save(self.teacher_model.state_dict(), path)
        else:
            torch.save(self.student_model.state_dict(), path)

        torch.save(self.optimizer.state_dict(), opt_path)
        self.cleanup("model_bsrnn", self.save_model_dir, extension='ckpt'+'.best', nkeep=3)
        self.cleanup("model_bsrnn", self.save_model_dir, extension='ckpt', nkeep=2)
        self.cleanup("opt_bsrnn", self.save_model_dir, extension='ckpt', nkeep=1)

    def cleanup(self, name1: str, dirname: str, extension: str, nkeep=5):
        if nkeep < 0:
            return
        checkpoints = glob.glob(os.path.join(dirname, f"{name1}*.{extension}"))
        if len(checkpoints) == 0:
            return
        checkpoints = sorted(checkpoints, key=self.get_epoch, reverse=True)
        for cp in checkpoints[nkeep:]:
            logger.debug("Removing old checkpoint: {}".format(cp))
            os.remove(cp)

    def get_epoch(self, cp) -> int:
        return int(os.path.basename(cp).split(".")[0].split("_")[-1])

    def is_clipped(self, audio, clipping_threshold=0.99):
        return any(abs(audio) > clipping_threshold)

    def segmental_snr_mixer(self, clean_bst, noise_bst, snr=None):
        # shuffle noise in batch
        noise_bst = noise_bst[torch.randperm(noise_bst.size(0))]
        if snr is None:
            snr =  random.randint(-5, 5)
        rmsclean = torch.sqrt(torch.mean(torch.square(clean_bst), dim=1))
        rmsnoise = torch.sqrt(torch.mean(torch.square(noise_bst), dim=1))

        # noisescalar
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noisenewlevel = noise_bst * noisescalar.reshape(-1, 1).repeat(1, noise_bst.size(1))
        noisy_bst = clean_bst + noisenewlevel
        noisy_absmax_values = torch.max(noisy_bst.abs(), dim=1).values
        clean_absmax_values = torch.max(clean_bst.abs(), dim=1).values
        max_values, _ = torch.max(torch.stack((noisy_absmax_values, clean_absmax_values), dim=0), dim=0)

        max_values[max_values<0.99] = 1
        noisy_bst = noisy_bst / max_values.reshape(-1, 1).repeat(1, noise_bst.size(1))
        clean_bst = clean_bst / max_values.reshape(-1, 1).repeat(1, noise_bst.size(1))
        return noisy_bst, clean_bst

    @torch.no_grad()
    def infence(self, batch, model):
        noisy = batch['mix'].to(self.dev)
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                onesided=True,return_complex=True)
        if model == "teacher":
            self.teacher_model.eval()
            teacher_model_load = torch.nn.DataParallel(self.teacher_model, device_ids=self.gpuid)
            est_spec = torch.view_as_complex(teacher_model_load(noisy_spec))
            est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                            onesided =True)
        elif model == "student":
            self.student_model.eval()
            student_model_load = torch.nn.DataParallel(self.student_model, device_ids=self.gpuid)
            est_spec = torch.view_as_complex(student_model_load(noisy_spec))
            est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                            onesided =True)
        return est_audio
        
    def train_step(self, noisy_spec_bst, clean_spec_bst):
        self.student_model.train()
        self.optimizer.zero_grad()
        student_model_load = torch.nn.DataParallel(self.student_model, device_ids=self.gpuid)
        est_spec_bst = torch.view_as_complex(student_model_load(noisy_spec_bst))
        weigeted_loss = {}
        loss, sl_loss, mrsl_loss = self.loss(clean_spec_bst, est_spec_bst)
        weigeted_loss['sl_loss'] = sl_loss.detach()
        weigeted_loss['mrsl_loss'] = mrsl_loss.detach()
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=5)
        self.optimizer.step()
        return loss, weigeted_loss

    def train(self, epoch, max_epoch):
        best_loss = 1000
        # no_impr = 0
        self.scheduler.best = best_loss       
        for epoch in range(epoch, max_epoch):
            concatenated_est_audio = np.empty((0, 48000), dtype='float32') 
            concatenated_mix_audio = np.empty((0, 48000), dtype='float32') 
            predicted_p808_mos = []
        
            for idx, batch in enumerate(self.train_ds):
                est_audio = self.infence(batch, "teacher").cpu().numpy()
                ratio_of_ones_all = Parallel(n_jobs=-1)(delayed(compute_vad)(audio) for audio in est_audio)
                ratio_of_ones_all = np.array(ratio_of_ones_all)

                predicted_p808_mos_averg = self.compute_score(est_audio, is_mean=False) * ratio_of_ones_all
                assert len(ratio_of_ones_all) == len(predicted_p808_mos_averg)
                predicted_p808_mos.append(predicted_p808_mos_averg)
                concatenated_est_audio = np.concatenate((concatenated_est_audio, est_audio), axis=0)
                concatenated_mix_audio = np.concatenate((concatenated_mix_audio, batch['mix'].numpy()), axis=0)
            predicted_p808_mos_vector = np.concatenate(predicted_p808_mos)
            sorted_indices = np.argsort(predicted_p808_mos_vector)[::-1]
            # top50%
            est_audio_top50_percent = concatenated_est_audio[sorted_indices[:int(len(concatenated_est_audio)*0.5)]]
            mix_audio_top50_percent = concatenated_mix_audio[sorted_indices[:int(len(concatenated_est_audio)*0.5)]]
            del concatenated_est_audio, concatenated_mix_audio
            gc.collect()

            cur_model_lr = self.optimizer.param_groups[0]["lr"]
            logger.info("Start train(student) epoch {} with batch size {} | opt_bsrnn_lr {:.3e}".format(epoch, self.train_bs, cur_model_lr))   
            loss_mem = []
            weigeted_loss_epoch = defaultdict(list)
            step = 0
            for s in range(0, len(est_audio_top50_percent) - 16 + 1, 16):
                batch_est = torch.tensor(est_audio_top50_percent[s:s + 16]).to(self.dev)
                batch_mix = torch.tensor(mix_audio_top50_percent[s:s + 16]).to(self.dev)
                # Remix-it
                batch_noise = batch_mix - batch_est
                noisy_mbst, clean_mbst = self.segmental_snr_mixer(clean_bst=batch_mix, noise_bst=batch_noise)
                noisy_ebst, clean_ebst = self.segmental_snr_mixer(clean_bst=batch_est, noise_bst=batch_noise)
                noisy_spec = torch.stft(batch_mix, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                    onesided=True,return_complex=True)
                clean_spec = torch.stft(batch_est, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                        onesided=True,return_complex=True)
                noisy_spec_mbst = torch.stft(noisy_mbst, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                    onesided=True,return_complex=True)
                clean_spec_mbst = torch.stft(clean_mbst, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                        onesided=True,return_complex=True)
                noisy_spec_ebst = torch.stft(noisy_ebst, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                    onesided=True,return_complex=True)
                clean_spec_ebst = torch.stft(clean_ebst, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).to(self.dev),
                                        onesided=True,return_complex=True)
                noisy_spec_bst = torch.cat((noisy_spec, noisy_spec_mbst, noisy_spec_ebst), dim=0)
                clean_spec_bst = torch.cat((clean_spec, clean_spec_mbst, clean_spec_ebst), dim=0)
                loss, weigeted_loss = self.train_step(noisy_spec_bst, clean_spec_bst)
                loss_mem.append(loss)
                weigeted_loss_epoch['SpecLoss'].append(weigeted_loss['sl_loss'])
                weigeted_loss_epoch['MRSpecLoss'].append(weigeted_loss['mrsl_loss'])
                step += 1
                if (step % self.log_freq) == 0:
                    loss_mean = torch.stack(loss_mem[-self.log_freq:]).mean().cpu()
                    loss_dict = {"loss": loss_mean.item()}
                    log_metrics(f"[{epoch}/{str(step)}]", loss_dict)
            metrics_train = {"loss": torch.stack(loss_mem).mean().cpu().item()}
            metrics_train.update({n: torch.mean(torch.stack(vals)).item() for n, vals in weigeted_loss_epoch.items()})
            log_metrics(f"[{epoch}] [train end(student)]", metrics_train)
            teacher_p808_mos = 0
            student_p808_mos = 0
            logger.info("begin competitive alternation in epoch {} with batch size {}".format(epoch, self.train_bs)) 
            for idx_t, batch in enumerate(self.valid_ds):
                est_audio = self.infence(batch, "teacher").cpu().numpy()
                dns_dict = self.compute_score(est_audio)
                teacher_p808_mos += dns_dict['p808_mos']
            teacher_p808_mos = teacher_p808_mos / (idx_t+1)
            for idx_s, batch in enumerate(self.valid_ds):
                est_audio = self.infence(batch, "student").cpu().numpy()
                dns_dict = self.compute_score(est_audio)
                student_p808_mos += dns_dict['p808_mos']
            student_p808_mos = student_p808_mos / (idx_s+1)
            if teacher_p808_mos >= student_p808_mos:
                logger.info("continue train | teacher p808_mos: {:.5f} | student p808_mos: {:.5f}".format(teacher_p808_mos, student_p808_mos))
                self.save_checkpoint(epoch=epoch, save_teacher=False)   
            else:
                logger.info("revert train | teacher p808_mos: {:.5f} | student p808_mos: {:.5f}".format(teacher_p808_mos, student_p808_mos)) 
                teacher_params = copy.deepcopy(self.teacher_model.state_dict())
                self.teacher_model.load_state_dict(self.student_model.state_dict())
                self.student_model.load_state_dict(teacher_params)
                logger.info(f"Writing better teacher model {self.save_model_dir} at epoch {epoch}")
                self.save_checkpoint(epoch=epoch, suffix="best")
            # schedule
            self.scheduler.step(metrics_train["loss"])
        logger.info(f"Writing the last student model in {self.save_model_dir} with epoch {epoch}")
        self.save_checkpoint(epoch=epoch, suffix="latest", save_teacher=False)
        logger.info("Finished training")  

if __name__=="__main__":
    main()
