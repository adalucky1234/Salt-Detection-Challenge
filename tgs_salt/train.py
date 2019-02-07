import numpy as np
def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train and Predict tgs salt Kaggle challenge')
    parser.add_argument('-t', action="store", dest="train", default="/mnt/image_translation/data/train/", help='Location of train data', type = str)
    parser.add_argument('-x', action="store", dest="test", default="/mnt/image_translation/data/test/", help='Location of test data', type = str)
    parser.add_argument('-s', action="store", dest="save", default="/results/trained_models/", help='Location to save trained model', type = str)
    parser.add_argument('-lr', action="store", dest="learning_rate", default=1e-4, help='Location to save trained model', type =float)
    parser.add_argument('-e', action="store", dest="num_epochs", default=30, help='Number of epochs to train', type =int)
    parser.add_argument('-bs', action="store", dest="num_epochs", default=32, help='Batch size', type =int)
    args = parser.parse_args()
    return args

def split_train_val(ids, train_val_ratio=0.8):
    assert(train_val_ratio <= 1.0 and train_val_ratio >0.0)


    import random
    num_total = len(ids)
    print("Total number of training and validation examples = {}".format(str(num_total)))

    train_boundary = int(num_total*train_val_ratio)
    shuffled_ids = ids
    random.shuffle(shuffled_ids)
    
    return shuffled_ids[0:train_boundary], shuffled_ids[train_boundary:]



def get_ids(path):
    """ Given a path get all the image ids associated with that folder
    """
    import glob
    image_filenames = glob.glob(path+"*.png")
    return [im.split('/')[-1][:- 4] for im in image_filenames]

def get_inputs_and_labels(ids, path, args):
    from utils.io_utils import ids_to_ndarrays

    train_ids, val_ids = split_train_val(ids)
    train_inputs = ids_to_ndarrays(train_ids, path+"images/", True,args)
    train_labels = ids_to_ndarrays(train_ids, path+"masks/", False,args)
    val_inputs = ids_to_ndarrays(val_ids, path+"images/", True,args)
    val_labels = ids_to_ndarrays(val_ids, path+"masks/", False,args)
    return train_inputs, train_labels, val_inputs, val_labels, val_ids

def get_test_inputs(path, args):
    from utils.io_utils import ids_to_ndarrays
    ids = get_ids(path+"images/")
    test_inputs = ids_to_ndarrays(ids, path+"images/", True, args)
    return test_inputs, ids


def train(ids, args):
    import tensorflow as tf
    from model.unetModel import UnetModel as Model
    from model.unetModel import Config
    config = Config()
    train_inputs, train_labels, val_inputs, val_labels, val_ids = get_inputs_and_labels(ids, args.train, args)
    train_inputs = np.append(train_inputs, [np.fliplr(x) for x in train_inputs], axis=0)
    train_labels = np.append(train_labels, [np.fliplr(x) for x in train_labels], axis=0)

    print(train_inputs.shape)
    print(train_labels.shape)
    with tf.Graph().as_default():
    # Build the model and add the variable initializer Op
        #from model.squeezenetModel import SqueezenetModel, Config
        model =  Model(config)
        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            losses  = model.fit(sess, train_inputs, train_labels, val_inputs, val_labels)
            test_inputs, test_ids  = get_test_inputs(args.test, args)
            test_ids, labels  = model.predict(sess, test_inputs, ids =test_ids)
            #val_ids, labels  = model.predict(sess, val_inputs, ids =val_ids)




def main():
    args = parse()
    ids = get_ids(args.train+"images/")
    train(ids, args)


if __name__ == '__main__':
    main()
