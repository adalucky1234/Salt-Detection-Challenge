
import tensorflow as tf

def batch_activate(x):
    """
    A batch norm layers followed by ReLU actviation
    """
    batch_norm = tf.layers.batch_normalization(inputs=x)
    return tf.nn.relu(features=batch_norm)

def convolution_block(x,
                        filters,
                        size,
                        strides=(1,1),
                        padding='same',
                        actviation = True
                        ):
    """
    A convolution block followed by a batch actviation block

    """
    conv_output = tf.layers.conv2d(input=x, kernel_size=size, filters=filters, strides=strides, padding=padding)
    if(actviation):
        return batch_activate(conv_output)
    return conv_output

def residual_block(
                    blockInput,
                    numFilters=16,
                    batchActivation=False):
    """
    A residual block with optional batch norma and activation at the end
    """
    x = batch_activate(blockInput)



