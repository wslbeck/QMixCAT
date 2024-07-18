from typing import Dict, Optional, Tuple
from loguru import logger
from collections import defaultdict
from torch.types import Number
import sys
import os
import torch
from socket import gethostname
import subprocess
from copy import deepcopy

from libs.utils import get_device


_logger_initialized = False
WARN_ONCE_NO = logger.level("WARNING").no + 1
DEPRECATED_NO = logger.level("WARNING").no + 2


def get_host() -> str:
    return gethostname()

def get_commit_hash():
    """Returns the current git commit."""
    try:
        git_dir = get_git_root()
        if git_dir is None:
            return None
        args = ["git", "-C", git_dir, "rev-parse", "--short", "--verify", "HEAD"]
        return subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        return None

def get_git_root():
    """Returns the top level git directory or None if not called within the git repository."""
    try:
        git_local_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["git", "-C", git_local_dir, "rev-parse", "--show-toplevel"]
        return subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        return None
    
def get_branch_name():
    try:
        git_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["git", "-C", git_dir, "rev-parse", "--abbrev-ref", "HEAD"]
        branch = subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        branch = None
    return branch

def init_logger(file: Optional[str] = None, level: str = "INFO", model: Optional[str] = None):
    global _logger_initialized, _duplicate_filter
    if _logger_initialized:
        logger.debug("Logger already initialized.")
    else:
        logger.remove()
        level = level.upper()
        if level.lower() != "none":
            log_format = Formatter(debug=logger.level(level).no <= logger.level("DEBUG").no).format
            logger.add(
                sys.stdout,
                level=level,
                format=log_format,
                filter=lambda r: r["level"].no not in {WARN_ONCE_NO, DEPRECATED_NO},
            )
            if file is not None:
                logger.add(
                    file,
                    level=level,
                    format=log_format,
                    filter=lambda r: r["level"].no != WARN_ONCE_NO,
                )

            logger.info(f"Running on torch {torch.__version__}")
            logger.info(f"Running on host {get_host()}")
            commit = get_commit_hash()
            if commit is not None:
                logger.info(f"Git commit: {commit}, branch: {get_branch_name()}")
            jobid = os.getenv("SLURM_JOB_ID")
            if jobid is not None:
                logger.info(f"Slurm jobid: {jobid}")
            logger.level("WARNONCE", no=WARN_ONCE_NO, color="<yellow><bold>")
            logger.add(
                sys.stderr,
                level=max(logger.level(level).no, WARN_ONCE_NO),
                format=log_format,
                filter=lambda r: r["level"].no == WARN_ONCE_NO and _duplicate_filter(r),
            )
            logger.level("DEPRECATED", no=DEPRECATED_NO, color="<yellow><bold>")
            logger.add(
                sys.stderr,
                level=max(logger.level(level).no, DEPRECATED_NO),
                format=log_format,
                filter=lambda r: r["level"].no == DEPRECATED_NO and _duplicate_filter(r),
            )
    if model is not None:
        logger.info("Loading model settings of {}", os.path.basename(model.rstrip("/")))
    _logger_initialized = True

class Formatter:
    def __init__(self, debug=False):
        if debug:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
                " | <level>{message}</level>"
            )
        else:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>BSRNN</cyan>"
                " | <level>{message}</level>"
            )
        self.fmt += "\n{exception}"

    def format(self, record):
        if record["level"].no == WARN_ONCE_NO:
            return self.fmt.replace("{level: <8}", "WARNING ")
        return self.fmt
    

def log_metrics(prefix: str, metrics: Dict[str, Number], level="INFO"):
    msg = ""
    stages = defaultdict(str)
    # loss_msg = ""
    for n, v in sorted(metrics.items(), key=_metrics_key):
        if abs(v) > 1e-3:
            m = f" | {n}: {v: #.5f}"
        else:
            m = f" | {n}: {v: #.3E}"
        msg += m
    if len(msg) > 0:
        logger.log(level, prefix + msg)
    
def _metrics_key(k_: Tuple[str, float]):
    k0 = k_[0]
    ks = k0.split("_")
    if len(ks) > 2:
        try:
            return int(ks[-1])
        except ValueError:
            return 1000
    elif k0 == "loss":
        return -999
    elif "loss" in k0.lower():
        return -998
    elif k0 == "lr":
        return 998
    elif k0 == "wd":
        return 999
    else:
        return -101

def log_model_summary(model: torch.nn.Module, dev, n_fft: int, n_hop: int):
    try:
        import ptflops
    except ImportError:
        logger.debug("Failed to import ptflops. Cannot print model summary.")
        return

    b = 1
    t = 16000
    device = dev

    noisy_td = torch.randn([b, t]).to(device)
    # aux_emb = torch.randn([b, 1, 256]).to(device)
    noisy_spec = torch.stft(noisy_td, n_fft, n_hop, window=torch.hann_window(n_fft).to(device),
                                onesided=True,return_complex=True)
    noisy_spec = torch.view_as_real(noisy_spec)
    B, F, T, C = noisy_spec.shape
    macs, params = ptflops.get_model_complexity_info(deepcopy(model.to(device)), (B, F, T, C), input_constructor=lambda _: {'x': noisy_spec}, as_strings=False,
        print_per_layer_stat=False, verbose=False)
    logger.info(f"Model complexity: {params/1e6:.3f}M #Params, {macs/1e9:.1f}G MACS")