"""specdec: a from-scratch tokenizer + speculative-decoding inference engine.

    from specdec.tokenizer import BPETokenizer
    from specdec.model import GPT, GPTConfig
    from specdec.generate import generate_naive, generate_speculative

See the package README for what's exact, what's benchmarked, and what a
from-scratch clone of this genre doesn't usually bother implementing.
"""

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

__all__ = ["GPT", "GPTConfig", "BPETokenizer"]
__version__ = "0.1.0"
