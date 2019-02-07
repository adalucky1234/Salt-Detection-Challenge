import time

import numpy as np
import tensorflow as tf

from model.model import Model
from utils.general_utils import get_minibatches
import tensorflow.contrib.layers as layers
from utils.general_utils import Progbar


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    IMAGE_HEIGHT = 101
    IMAGE_WIDTH = 101
    NUM_CHANNELS = 3
    n_epochs = 30
    lr = 1e-4
    batch_size = 32


class ResnetModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        # YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.float32, shape=(
            None,self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.NUM_CHANNELS), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(
            None, 10201), name="labels")
        # END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # YOUR CODE HERE
        if labels_batch is not None:
            feed_dict = {self.input_placeholder: inputs_batch,
                         self.labels_placeholder: labels_batch}
        else:
            feed_dict = {self.input_placeholder: inputs_batch}

        # END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:

        y = softmax(Wx + b)

        Hint: Make sure to create tf.Variables as needed.
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.

        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        # YOUR CODE HERE
        import tensorflow.contrib.layers as layers
        conv7 = layers.conv2d(inputs=self.input_placeholder,num_outputs=32,kernel_size=7,stride=1,padding='VALID',activation_fn=tf.nn.relu)
        spatialBN_layer = layers.batch_norm(inputs=conv7,center=True,scale=True,trainable=True)
        spatialMP_layer = layers.max_pool2d(inputs=spatialBN_layer,kernel_size=2,stride=2,padding='VALID')
        flat = layers.flatten( spatialMP_layer)    
        affine = layers.fully_connected(inputs=flat,num_outputs=256,activation_fn=tf.nn.relu)
        self.preds = layers.fully_connected(inputs=affine,num_outputs=10201,activation_fn=None)
        """
        initializer = tf.contrib.layers.xavier_initializer()
        
        # Create a dictionary which holds the output of the 
        # residual blocks
        res_block = {}
        
        # Specify the number of filters in each residual block
        filters = [32, 64, 128, 256, 512]
        is_training=True
        
        # First convolution of the network
        first_conv = tf.layers.conv2d(self.input_placeholder, 32, 7, padding='same', activation=None, 
                                      kernel_initializer=initializer,
                                      name='First_convolution')
        print("First conv size {}".format(str(first_conv.get_shape())))
        batchnorm_layer = tf.layers.batch_normalization(first_conv, training=is_training,
                                                       name='First_batchnorm')
        print("Batchnorm layer 0 size {}".format(str(batchnorm_layer.get_shape())))
        relu_layer = tf.nn.relu(batchnorm_layer, name='First_relu')
        print("Relu layer 0 size {}".format(str(relu_layer.get_shape())))
        
        for i in range(5):
            # Residual block
            res_block[i] = residual_block(relu_layer, is_training, filters[i], 3, 5, i,
                                          initializer, reduce_size=False)
            if i!=4:
                # Reduce 2D dimension
                strided_conv = tf.layers.conv2d(res_block[i], filters[i+1], 3, strides=2, 
                                                padding='same', activation=None, 
                                                kernel_initializer=initializer,
                                                name='Strided_convolution_%d' % i)
                print("Strided conv layer {} size {}".format(str(i), str(strided_conv.get_shape())))
                batchnorm_strided = tf.layers.batch_normalization(strided_conv, 
                                                                  training=is_training,
                                                                  name='Batchnorm_strided_%d' % i)
                print("Batchnorm layer {} size {}".format(str(i), str(batchnorm_strided.get_shape())))
                relu_layer = tf.nn.relu(batchnorm_strided, name='Relu_%d' % i)
                print("Relu layer {} size {}".format(str(i), str(relu_layer.get_shape())))
        
        # Last convolution
        last_conv = tf.layers.conv2d(res_block[4], 256, 1, padding='valid', 
                                     activation=None, kernel_initializer=initializer,
                                     name='Last_convolution')
        #print("Last conv size {}".format(str(last_conv.get_shape())))
        #last_batchnorm = tf.layers.batch_normalization(last_conv, training=is_training,
        #                                           name='Last_batchnorm')
        #print("Last Batchnorm layer size {}".format(str(last_batchnorm.get_shape())))
        flat = tf.layers.flatten(relu_layer)
        affine = tf.contrib.layers.fully_connected(inputs=flat, num_outputs = 12544, activation_fn= tf.nn.relu)    
        self.preds = tf.contrib.layers.fully_connected(inputs=affine,num_outputs=10201,activation_fn=None)
        print("Preds shape {}".format(str(self.preds.get_shape())))
        # END YOUR CODE
        return self.preds

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        # YOUR CODE HERE
        temp = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=self.labels_placeholder)
        loss = tf.reduce_mean(temp)
        # END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.
ixers/
        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        # YOUR CODE HERE
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        # END YOUR CODE
        return train_op

    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        inputs_shape = inputs.shape
        prog = Progbar(target=1 + inputs_shape[0] / self.config.batch_size)
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            loss = self.train_on_batch(sess, input_batch, labels_batch)
            total_loss += loss
            prog.update(n_minibatches, [("train loss", loss)])
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            #duration = time.time() - start_time
            #print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses


    def save_model(self, saver):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        feed_dict = self.create_feed_dict(inputs_batch)
        preds = sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def predict(self, sess, inputs, ids):
        labels = []
        for inputs_batch in get_minibatches(inputs, self.config.batch_size):
            raw_preds = self.predict_on_batch(sess, inputs_batch)
            masks = self.convert_to_mask(raw_preds, 0.0)
            labels += masks
        return ids,labels

    def convert_to_mask(self,preds, threshold):
        from utils.data_utils import mask_to_output
        mask = (preds> threshold).astype(int)
        return mask_to_output(mask)

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

def normal_residual_unit(input_layer, is_training, filters,
                         kernel_size, initializer, num_unit):
    '''
    Residual unit as in https://arxiv.org/pdf/1603.05027.pdf.
    '''
    with tf.variable_scope('Residual_unit_%d' % num_unit):
        first_batch = tf.layers.batch_normalization(input_layer, training=is_training,
                                                    name='First_batchnorm')
        first_relu = tf.nn.relu(first_batch, name='First_relu')
        first_conv = tf.layers.conv2d(first_relu, filters, kernel_size, padding='same', 
                                      activation=None, kernel_initializer=initializer,
                                      name='First_convolution')
        sec_batch = tf.layers.batch_normalization(first_conv, training=is_training,
                                                  name='Second_batchnorm')
        sec_relu = tf.nn.relu(sec_batch, name='Second_relu')
        sec_conv = tf.layers.conv2d(sec_relu, filters, kernel_size, padding='same',
                                    activation=None, kernel_initializer=initializer,
                                    name='Second_convolution')        
        addition = input_layer+sec_conv
    return addition

def reshaped_residual_unit(input_layer, is_training, filters,
                           kernel_size, initializer, num_unit):
    '''
    Residual unit as in https://arxiv.org/pdf/1603.05027.pdf.
    To account for the difference in dimensions:
    - the filter size is downsampled by a 1x1 convolution
      with a stride of 2.
    - the number of filters is increased with the same
      1x1 convolution.
    '''
    with tf.variable_scope('Residual_unit_%d' % num_unit):
        first_batch = tf.layers.batch_normalization(input_layer, training=is_training,
                                                   name='First_batchnorm')
        first_relu = tf.nn.relu(first_batch, name='First_relu')
        first_conv = tf.layers.conv2d(first_relu, filters, kernel_size, strides=2,
                                      padding='same', activation=None, 
                                      kernel_initializer=initializer,
                                      name='First_convolution')
        sec_batch = tf.layers.batch_normalization(first_conv, training=is_training,
                                                  name='Second_batchnorm')
        sec_relu = tf.nn.relu(sec_batch, name='Second_relu')
        sec_conv = tf.layers.conv2d(sec_relu, filters, kernel_size, padding='same',
                                    activation=None, kernel_initializer=initializer,
                                    name='Second_convolution')        
        reshaped_inp = tf.layers.conv2d(input_layer, filters, 1, strides=2, 
                                       activation=None, kernel_initializer=initializer,
                                       name='Reshaped_input')
        addition = reshaped_inp+sec_conv
    return addition

def residual_block(input_layer, is_training, filters, kernel_size, 
                   num_units, num_block, initializer, reduce_size=True):
    '''
    Builds a block of N residual units.
    '''
    res_units = {}
    with tf.variable_scope('Residual_block_%d' % num_block):
        if reduce_size:
            res_units[0] = reshaped_residual_unit(input_layer, is_training,
                filters, kernel_size, initializer, 0)
        else:
            res_units[0] = normal_residual_unit(input_layer, is_training,
                filters, kernel_size, initializer, 0)
        for i in range(1, num_units):
            res_units[i] = normal_residual_unit(res_units[i-1], is_training,
                filters, kernel_size, initializer, i)
    return res_units[num_units-1]

