from .encoder import Encoder
from .encoder_rlsplat import EncoderRLSplat, EncoderRLSplatCfg

ENCODERS = {
    "rlsplat": EncoderRLSplat,
}

EncoderCfg = EncoderRLSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, None]:
    return ENCODERS[cfg.name](cfg), None
