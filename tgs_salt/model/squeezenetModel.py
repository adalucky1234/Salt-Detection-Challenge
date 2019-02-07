import time

import numpy as np
import tensorflow as tf

from model.model import Model
from utils.general_utils import get_minibatches
import tensorflow.contrib.layers as layers


def fire_module(x,inp,sp,e11p,e33p):
    with tf.variable_scope("fire"):
        with tf.variable_scope("squeeze"):
            W = tf.get_variable("weights",shape=[1,1,inp,sp])
            b = tf.get_variable("bias",shape=[sp])
            s = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")+b
            s = tf.nn.relu(s)
        with tf.variable_scope("e11"):
            W = tf.get_variable("weights",shape=[1,1,sp,e11p])
            b = tf.get_variable("bias",shape=[e11p])
            e11 = tf.nn.conv2d(s,W,[1,1,1,1],"VALID")+b
            e11 = tf.nn.relu(e11)
        with tf.variable_scope("e33"):
            W = tf.get_variable("weights",shape=[3,3,sp,e33p])
            b = tf.get_variable("bias",shape=[e33p])
            e33 = tf.nn.conv2d(s,W,[1,1,1,1],"SAME")+b
            e33 = tf.nn.relu(e33)
        return tf.concat([e11,e33],3)



class Config(object):
    """Holds model hyperparams and da ta information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    IMAGE_HEIGHT = 64
    IMAGE_WIDTH = 64
    NUM_CHANNELS = 3
    NUM_CLASSES = 2
    n_epochs = 100
    lr = 1e-4
    batch_size = 64


class SqueezenetModel(Model):
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
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(
            None), name="labels")
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
        """
        # YOUR CODE HERE
        """
        import tensorflow.contrib.layers as layers
        conv7 = layers.conv2d(inputs=self.input_placeholder,num_outputs=32,kernel_size=7,stride=1,padding='VALID',activation_fn=tf.nn.relu)
        spatialBN_layer = layers.batch_norm(inputs=conv7,center=True,scale=True,trainable=True)
        spatialMP_layer = layers.max_pool2d(inputs=spatialBN_layer,kernel_size=2,stride=2,padding='VALID')
        flat = layers.flatten( spatialMP_layer)    
        affine = layers.fully_connected(inputs=flat,num_outputs=256,activation_fn=tf.nn.relu)
        pred = layers.fully_connected(inputs=affine,num_outputs=2,activation_fn=None)
        """
        init = tf.contrib.layers.xavier_initializer()
        
        # Create a dictionary which holds the output of the 
        # residual blocks
        self.layers= []
        x = self.input_placeholder
        self.layers = self.extract_features(x, reuse=False)
        self.features = self.layers[-1]
        with tf.variable_scope('classifier'):
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            with tf.variable_scope('layer1'):
                W = tf.get_variable("weights",shape=[1,1,512,2], initializer= init)
                b = tf.get_variable("bias",shape=[2], initializer=init)
                x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
                x = tf.nn.bias_add(x,b)
                self.layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            with tf.variable_scope('layer3'):
                x = tf.contrib.layers.avg_pool2d(x,[1,1],stride=1,padding='VALID')
                self.layers.append(x)
        self.preds = tf.reshape(x,[-1, self.config.NUM_CLASSES])
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
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=tf.one_hot(self.labels_placeholder,2)))
        # END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.
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

    def extract_features(self, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('layer0'):
                W = tf.get_variable("weights",shape=[3,3,3,64])
                b = tf.get_variable("bias",shape=[64])
                x = tf.nn.conv2d(x,W,[1,2,2,1],"VALID")
                x = tf.nn.bias_add(x,b)
                layers.append(x)
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer3'):
                x = fire_module(x,64,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer4'):
                x = fire_module(x,128,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer6'):
                x = fire_module(x,128,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer7'):
                x = fire_module(x,256,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer9'):
                x = fire_module(x,256,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer10'):
                x = fire_module(x,384,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer11'):
                x = fire_module(x,384,64,256,256)
                layers.append(x)
            with tf.variable_scope('layer12'):
                x = fire_module(x,512,64,256,256)
                layers.append(x)
        return layers


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
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
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
            duration = time.time() - start_time
            print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses




    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

