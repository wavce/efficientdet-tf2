from core.layers import build_activation
from core.layers import build_normalization


def prediction_head(inputs,
                    shared_convolutions,
                    normalization,
                    activation="swish",
                    repeats=3,
                    level=3,
                    name="prediction_head"):
    x = inputs
    for i in range(repeats):
        x = shared_convolutions[i](x)
        if normalization is not None:
            x = build_normalization(normalization=normalization.normalization,
                                    name=name+"-%d-bn-%d" % (i, level))(x)

        if activation is not None:
            x = build_activation(activation=activation.activation, 
                                 name=name+"-%d-%s-%d" % (i, activation.activation, level))(x)

    return x

    

