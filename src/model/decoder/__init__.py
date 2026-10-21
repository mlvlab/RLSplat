from .decoder import Decoder, DecoderSplattingCUDACfg

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    if decoder_cfg.name != "splatting_cuda":
        raise ValueError(f"Unknown decoder: {decoder_cfg.name}")

    from .decoder_splatting_cuda import DecoderSplattingCUDA

    return DecoderSplattingCUDA(decoder_cfg)
