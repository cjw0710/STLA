"""Environment-conditioned components for the WWW 2027 prototype."""

from .dynamic_low_rank_mask import DynamicLowRankMask
from .environment_encoder import EnvironmentEncoder, build_environment_features
from .sparse_propagation import SparseGraphPropagation
from .temporal_diffusion import TemporalDiffusionModel
from .temporal_logit_adapter import TemporalLogitAdapter
from .dynamic_dediff import DynamicDeDiff, EnvironmentLowRankDebiasing
from .paper_graph_disentangler import (
    PaperGCNEncoder,
    PaperGraphDisentangler,
    PaperGraphOutput,
    PaperLowRankEdgeMask,
    SparseGraphSplit,
)
from .paper_faithful_dediff import (
    PaperDeDiffLoss,
    PaperDeDiffOutput,
    PaperFaithfulDeDiff,
    PaperSTAN,
    paper_dediff_loss,
)

__all__ = [
    "DynamicLowRankMask",
    "EnvironmentEncoder",
    "SparseGraphPropagation",
    "TemporalDiffusionModel",
    "TemporalLogitAdapter",
    "DynamicDeDiff",
    "EnvironmentLowRankDebiasing",
    "PaperGCNEncoder",
    "PaperGraphDisentangler",
    "PaperGraphOutput",
    "PaperLowRankEdgeMask",
    "SparseGraphSplit",
    "PaperDeDiffLoss",
    "PaperDeDiffOutput",
    "PaperFaithfulDeDiff",
    "PaperSTAN",
    "paper_dediff_loss",
    "build_environment_features",
]
