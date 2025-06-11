# from .VAE.vae_2d_resnet import VAERes2D
from .VAE.rqvae_2d_resnet import RQVAERes2D
from .VAE.quantizer_multi import VectorQuantizer
from .VAE.residual_quantizer import ResidualVectorQuantizer
from .VAE.residual_quantizer_v2 import ResidualVectorQuantizerV2
from .VAE.residual_quantizer_v3 import ResidualVectorQuantizerV3

from .transformer.pose_decoder import PoseDecoder
from .transformer.pose_encoder import PoseEncoder
# from .transformer.PlanUtransformer_multi import PlanUAutoRegTransformer
# from .transformer.PlanUtransformer_multi_decode import PlanUAutoRegTransformer
# from .transformer.hrt import HighResolutionTransformer
from .transformer.PlanUtransformer import PlanUAutoRegTransformer, PlanUAutoRegTransformerGuide, PlanUAutoRegTransformerResidual

from .TransVQVAE_3stage_wonhyeok import TransVQVAE
from .TransRQVAE_indiv import TransRQVAEIndiv
from .TransRQVAE_naive import TransRQVAENaive
from .TransRQVAE import TransRQVAE
from .TransRQVAE_rqvaealign import TransRQVAEalign
# from .TransVQVAE_var_clear import TransVQVAE
# from .transformer.var_pure  import VAR
# from .transformer.var_reformer import VAR
# from .transformer.var import VAR