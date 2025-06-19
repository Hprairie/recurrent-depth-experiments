import logging
import re

from recdep.recpre.tokenizer import Tokenizer
from recdep.recpre import optim
from recdep.recpre import settings
from recdep.recpre import monitor

# Inference
from recdep.recpre import raven_modeling_minimal

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["Tokenizer", "optim", "settings", "monitor", "raven_modeling_minimal"]
