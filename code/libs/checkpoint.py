from typing import List, Optional, Tuple, Union
import torch.nn as nn
from ..module import *
from loguru import logger


def init_model():
    return BSRNN(num_channel=64, num_layer=5), Discriminator(ndf=16)

def load_model(
    cp_dir: Optional[str],
    mask_only: bool = False,
    train_df_only: bool = False,
    extension: str = "ckpt",
    epoch: Union[str, int, None] = "latest",
) -> Tuple[nn.Module, int]:
    if mask_only and train_df_only:
        raise ValueError("Only one of `mask_only` `train_df_only` can be enabled")
    model_bsrnn, model_d = init_model()
    
    blacklist: List[str] = config("CP_BLACKLIST", [], Csv(), save=False, section="train")  # type: ignore
    if cp_dir is not None:
        epoch = read_cp(
            model, "model", cp_dir, blacklist=blacklist, extension=extension, epoch=epoch
        )
        epoch = 0 if epoch is None else epoch
    else:
        epoch = 0
    return model, epoch



def read_cp(
    obj: Union[torch.optim.Optimizer, nn.Module],
    name: str,
    dirname: str,
    epoch: Union[str, int, None] = "latest",
    extension="ckpt",
    blacklist=[],
    log: bool = True,
):
    checkpoints = []
    if isinstance(epoch, str):
        assert epoch in ("best", "latest")
    if epoch == "best":
        checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}.best"))
        if len(checkpoints) == 0:
            logger.warning("Could not find `best` checkpoint. Checking for default...")
    if len(checkpoints) == 0:
        checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}"))
        checkpoints += glob.glob(os.path.join(dirname, f"{name}*.{extension}.best"))
    if len(checkpoints) == 0:
        return None
    if isinstance(epoch, int):
        latest = next((x for x in checkpoints if get_epoch(x) == epoch), None)
        if latest is None:
            logger.error(f"Could not find checkpoint of epoch {epoch}")
            exit(1)
    else:
        latest = max(checkpoints, key=get_epoch)
        epoch = get_epoch(latest)
    if log:
        logger.info("Found checkpoint {} with epoch {}".format(latest, epoch))
    latest = torch.load(latest, map_location="cpu")
    latest = {k.replace("clc", "df"): v for k, v in latest.items()}
    if blacklist:
        reg = re.compile("".join(f"({b})|" for b in blacklist)[:-1])
        len_before = len(latest)
        latest = {k: v for k, v in latest.items() if reg.search(k) is None}
        if len(latest) < len_before:
            logger.info("Filtered checkpoint modules: {}".format(blacklist))
    if isinstance(obj, nn.Module):
        while True:
            try:
                missing, unexpected = obj.load_state_dict(latest, strict=False)
            except RuntimeError as e:
                e_str = str(e)
                logger.warning(e_str)
                if "size mismatch" in e_str:
                    latest = {k: v for k, v in latest.items() if k not in e_str}
                    continue
                raise e
            break
        for key in missing:
            logger.warning(f"Missing key: '{key}'")
        for key in unexpected:
            if key.endswith(".h0") or "erb_comp" in key:
                continue
            logger.warning(f"Unexpected key: {key}")
        return epoch
    obj.load_state_dict(latest)