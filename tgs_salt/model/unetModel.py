
import time, sys

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
    NUM_CHANNELS = 1
    n_epochs =  150
    lr = 1e-3
    batch_size = 32


class UnetModel(Model):
    """Implem ents a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Gen erates placeholder variables to represent the input tensors.

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
            None,self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 1), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(
            None, 101,101), name="labels")
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

        """
        from model.unet_utils import unet_downstream_block, unet_middle_block, unet_upstream_block
        initializer = tf.contrib.layers.xavier_initializer()

        print("Input block")
        print(self.input_placeholder.get_shape())
        
        first_block, first_conv = unet_downstream_block(self.input_placeholder, 16, 0.25)
        print("1st block")
        print(first_block.get_shape())
        second_block, second_conv = unet_downstream_block(first_block, 32, 0.5)
        print("2nd block")
        print(second_block.get_shape())
        third_block, third_conv = unet_downstream_block(second_block, 64, 0.5)
        print("3rd block")
        print(third_block.get_shape())
        fourth_block, fourth_conv = unet_downstream_block(third_block, 128, 0.5)
        print("Fourth block")
        print(fourth_block.get_shape())

        middle_block = unet_middle_block(fourth_block, 256)
        print("Middle block")
        print(middle_block.get_shape())

        first_upstream_block = unet_upstream_block(middle_block, 128, fourth_conv,0.5)
        print(first_upstream_block.get_shape())
        second_upstream_block = unet_upstream_block(first_upstream_block, 64, third_conv,0.5, padding = "valid")
        print(second_upstream_block.get_shape())
        third_upstream_block = unet_upstream_block(second_block, 32, second_conv,0.5)
        print(third_upstream_block.get_shape())
        fourth_upstream_block = unet_upstream_block(third_upstream_block, 16, first_conv,0.5, padding = "valid")
        print(fourth_upstream_block.get_shape())

        batch_size = tf.shape(fourth_upstream_block)[0]
        logits = tf.reshape(tf.layers.conv2d(inputs=fourth_upstream_block, kernel_size=(1,1), filters=1, strides=(1,1), padding="same"),[batch_size, 101,101])
        self.preds = tf.nn.sigmoid(logits)
        print(self.preds.get_shape())
        return logits#self.preds

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
        #temp = -tf.reduce_sum(self.labels_placeholder*tf.log(pred))
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
        print('\t')
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels, val_inputs=None, val_labels=None):
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
            if(epoch > 15):
                self.config.lr = 1e-4
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            losses.append(average_loss)
            score,score2, precision = self.evaluate(sess, val_inputs, val_labels)
            print('Epoch {:}: loss = {:.4f} IoUScore = {:.4f}, IoUscore2= {:.4f}, Precision = {:.4f} ({:.3f} sec)'.format(epoch, average_loss, score, score2, precision, duration))
        return losses

    def evaluate(self, sess, inputs, labels, metric=None):
        from utils.data_utils import Precision
        from utils.data_utils import get_iou_vector
        IoUs = []
        IoUs2 = []
        Precisions = []

        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            loss = self.train_on_batch(sess, input_batch, labels_batch)
            raw_preds = self.predict_on_batch(sess, input_batch)
            #IoUbatch = get_iou_vector(labels_batch.astype(int), (raw_preds>0.5).astype(int))
            #IoUs = IoUs + IoUbatch 
            pBatch, IoUbatch = Precision((raw_preds>0.5).astype(int), labels_batch) 
            Precisions = Precisions+ pBatch.tolist()
            IoUs = IoUs+IoUbatch.tolist()
            IoUs2 = IoUs2+get_iou_vector(labels_batch, (raw_preds>0.5).astype(int))
        return np.mean(IoUs), np.mean(np.asarray(IoUs2)), np.mean(np.mean(Precisions))


    def save_model(self, saver):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        feed_dict = self.create_feed_dict(inputs_batch)
        preds = sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def predict(self, sess, inputs, ids):
        labels = {}
        for ix, inputs_batch in enumerate(get_minibatches(inputs, self.config.batch_size, shuffle=False)):
            raw_preds = self.predict_on_batch(sess, inputs_batch)
            masks = self.convert_to_mask(raw_preds, 0.5)
            from utils.data_utils import rle_encode
            from PIL import Image
            import sys
            import numpy as np
            for kx, mask in enumerate(masks):
                #im = Image.fromarray(np.uint8(mask*255))
                #im.save("/results/preds/"+str(ids[ix+kx])+".png", "PNG")
                labels[ids[ix*self.config.batch_size+kx]] = rle_encode(mask)
            
        import pandas as pd
        df = pd.DataFrame.from_dict(labels, orient="index")
        df.index.names = ['id']
        df.columns = ['rle_mask']
        df.to_csv("../submission/results.csv")


        return ids,labels

    def convert_to_mask(self,preds, threshold):
        from utils.data_utils import mask_to_output
        mask = (preds> threshold).astype(int)
        return mask

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

