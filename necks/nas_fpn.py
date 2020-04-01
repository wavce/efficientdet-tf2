import enum
import tensorflow as tf
from core.layers import conv_block
from configs.params_dict import ParamsDict


COMBINATION_OPS = enum.Enum("COMBINATION_OPS", ["SUM", "GLOBAL_ATTENTION"])
NODE_TYPES = enum.Enum("NODE_TYPES", ["INTERMEDIATE", "OUTPUT"])


class Config(object):
    """NAS-FPN model config."""

    def __init__(self, model_config, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level
        self.nodes = self._parse(model_config)

    def _parse(self, config):
        """Parse model config from list of integer."""
        if len(config) % 4 != 0:
            raise ValueError("The length of node configs `{}` needs to be"
                             "divisible by 4.".format(len(config)))
        num_nodes = int(len(config) / 4)
        num_output_nodes = self.max_level - self.min_level + 1
        levels = list(range(self.max_level, self.min_level - 1, -1))

        nodes = []
        for i in range(num_nodes):
            node_type = NODE_TYPES.INTERMEDIATE if i < num_nodes - num_output_nodes else NODE_TYPES.OUTPUT

            level = levels[config[4*i]]
            combine_method = (COMBINATION_OPS.SUM if config[4*i + 1] == 0
                              else COMBINATION_OPS.GLOBAL_ATTENTION)
            input_offsets = [config[4*i + 2], config[4*i + 3]]
            nodes.append({
                "node_type": node_type,
                "level": level,
                "combine_method": combine_method,
                "input_offsets": input_offsets
            })

        return nodes


def resample_feature_map(x,
                         level,
                         target_level,
                         convolution="conv2d",
                         target_level_filters=256,
                         normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                         kernel_regularizer=None,
                         name="resample"):
    input_filters = tf.keras.backend.int_shape(x)[-1]
    if input_filters != target_level_filters:
        x = conv_block(convolution,
                       filters=target_level_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       kernel_regularizer=kernel_regularizer,
                       normalization=normalization,
                       activation=None,
                       name=name+"/"+convolution)(x)
    if level < target_level:
        strides = int(2 ** (target_level - level))
        x = tf.keras.layers.MaxPool2D(pool_size=strides,
                                      strides=strides,
                                      padding="same",
                                      name=name+"/pool")(x)
    elif target_level > level:
        size = int(2 ** (level - target_level_filters))
        x = tf.keras.layers.UpSampling2D(size=size, name=name+"/upsample")(x)

    return x


def global_attention(x1, x2, name="global_attention"):
    m = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True),
                               name=name+"/global_pooling")(x1)
    m = tf.keras.layers.Activation("sigmoid", name=name+"/sigmoid")(m)
    att = tf.keras.layers.Multiply(name=name+"/multiply")([x2, m])

    x2 = tf.keras.layers.Add(name=name+"/sum")([x1, att])

    return x2


def relu_conv2d_bn(x,
                   convolution,
                   filters=256,
                   kernel_size=(3, 3),
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                   kernel_regularizer=None,
                   activation="relu",
                   name="relu_conv2d_bn"):
    x = tf.keras.layers.Activation(activation, name=name+"/"+activation)(x)
    x = conv_block(convolution=convolution,
                   filters=filters,
                   kernel_size=kernel_size,
                   strides=(1, 1),
                   kernel_regularizer=kernel_regularizer,
                   normalization=normalization,
                   activation=None,
                   name=name)(x)
    return x


def nas_fpn(inputs,
            convolution="conv2d",
            normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
            activation=dict(activation="relu"),
            feat_dims=256,
            min_level=3,
            max_level=7,
            weight_decay=0.,
            dropblock=None,
            name="nas_fpn_neck"):
    normalization = normalization.as_dict() if isinstance(normalization, ParamsDict) else normalization
    activation = activation.as_dict() if isinstance(activation, ParamsDict) else activation
    kernel_regularizer = (tf.keras.regularizers.l2(weight_decay) 
                          if weight_decay is not None and weight_decay > 0 else None) 
    num_outputs = max_level - min_level + 1
    assert num_outputs == 5, "Only support 5 stage, i.e. P3, P4, P5, P6, P7."

    if min_level == 3 and max_level == 7:
        model_config = [
            3, 1, 1, 3,
            3, 0, 1, 5,
            4, 0, 0, 6,  # Output to level 3.
            3, 0, 6, 7,  # Output to level 4.
            2, 1, 7, 8,  # Output to level 5.
            0, 1, 6, 9,  # Output to level 7.
            1, 1, 9, 10]  # Output to level 6.
    else:
        raise ValueError("The NAS-FPN with min level {} and max level {} "
                         "is not supported.".format(min_level, max_level))
    config = Config(model_config, min_level, max_level)

    num_inputs = len(inputs)
    if num_inputs < num_outputs:
        features = []
        for i in range(num_inputs):
            feat = resample_feature_map(x=features[i],
                                        level=i,
                                        target_level=i,
                                        convolution=convolution,
                                        target_level_filters=feat_dims,
                                        kernel_regularizer=kernel_regularizer,
                                        name=name+"/resample_"+str(i))
            features.append(feat)
        for i in range(num_inputs, num_outputs):
            feat = resample_feature_map(x=features[-1],
                                        level=i,
                                        target_level=i,
                                        convolution=convolution,
                                        target_level_filters=feat_dims,
                                        normalization=normalization,
                                        name=name+"/resample_"+str(i))
            features.append(feat)
    else:
        features = inputs

    for i, sub_policy in enumerate(config.nodes):
        num_output_connections = [0] * len(features)
        new_level = sub_policy["level"]
        feature_levels = list(range(min_level, max_level+1))

        # Checks the range of input_offsets
        for input_offset in sub_policy["input_offsets"]:
            if input_offset >= len(features):
                raise ValueError(
                    "input_offset ({}) is larger than number of "
                    "features({})".format(input_offset, len(features)))
            input0 = sub_policy["input_offsets"][0]
            input1 = sub_policy["input_offsets"][1]

            # Update graph with inputs.
            node0 = features[input0]
            node0_level = feature_levels[input0]
            num_output_connections[input0] += 1
            node0 = resample_feature_map(x=node0,
                                         level=node0_level,
                                         target_level=new_level,
                                         convolution=convolution,
                                         target_level_filters=feat_dims,
                                         normalization=normalization,
                                         kernel_regularizer=kernel_regularizer,
                                         name=name+"/resample_node"+str(input0))

            node1 = features[input1]
            node1_level = feature_levels[input1]
            num_output_connections[input1] += 1
            node1 = resample_feature_map(x=node1,
                                         level=node1_level,
                                         target_level=new_level,
                                         convolution=convolution,
                                         target_level_filters=feat_dims,
                                         normalization=normalization,
                                         kernel_regularizer=kernel_regularizer,
                                         name=name+"resample_node"+str(input1))

            # Combine node0 and node1 to create new feature.
            if sub_policy["combine_method"] == COMBINATION_OPS.SUM:
                new_node = tf.keras.layers.Add(name=name+"/sum_node%d_node%d" % (input0, input1))([node0, node1])
            elif sub_policy["combine_method"] == COMBINATION_OPS.GLOBAL_ATTENTION:
                if node0_level >= node1_level:
                    new_node = global_attention(node0, node1,
                                                name=name+"/global_attention_node%d_node%d" % (node0_level, node1_level))
                else:
                    new_node = global_attention(node1, node0,
                                                name=name+"/global_attention_node%d_node%d" % (node1_level, node0_level))
            else:
                raise ValueError("Unknown combine_method {}.".format(sub_policy["combine_method"]))

            # Add intermediate nodes that do not have any connections to output
            if sub_policy["node_type"] == NODE_TYPES.OUTPUT:
                for j, (feat, feat_level, num_output) in enumerate(
                        zip(features, feature_levels, num_output_connections)):
                    if num_output == 0 and feat_level == new_level:
                        num_output_connections[j] += 1
                        new_node = tf.keras.layers.Add(name=name+"/sum_%d_%d" % (i, j))([new_node, feat])

            new_node = tf.keras.layers.Activation(activation, name=name+"/"+activation+"_"+str(i))(new_node)
            new_node = conv_block(convolution,
                                  filters=feat_dims,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=kernel_regularizer,
                                  normalization=normalization,
                                  activation=None,
                                  dropblock=dropblock,
                                  name=name+"/conv_"+str(i))(new_node)

            features.append(new_node)
            feature_levels.append(new_level)
            num_output_connections.append(0)

    outputs = []
    for i in range(len(features) - num_outputs, len(features)):
        # level = feature_levels[i]
        outputs.append(features[i])

    return outputs

