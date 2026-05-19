from src.encoding.rate import rate_encode
from src.encoding.latency import latency_encode
from src.encoding.delta import delta_encode

ENCODERS = {
    "rate": rate_encode,
    "latency": latency_encode,
    "delta": delta_encode,
}


def get_encoder(name: str):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoding {name!r}. Available: {list(ENCODERS)}")
    return ENCODERS[name]
