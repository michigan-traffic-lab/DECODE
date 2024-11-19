from domain_expansion.model.mtr.MTR import MotionTransformer
from domain_expansion.continual.hyperMTR import HyperMotionTransformer
from domain_expansion.continual.ewcMTR import EwcMotionTransformer
from domain_expansion.continual.erMTR import ErMotionTransformer

__all__ = {
    'pretrained':MotionTransformer,
    'DECODE': HyperMotionTransformer,
    'ewc':EwcMotionTransformer,
    'er':ErMotionTransformer,
    'no_rehearsal':MotionTransformer
}


def build_model(config):
    model = __all__[config.method.strategy_name](
        config=config
    )

    return model
