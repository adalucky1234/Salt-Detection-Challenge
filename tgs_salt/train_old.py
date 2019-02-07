











def generate_features_and_labels(image_path, mask_path): 
    import glob
    output_masks = glob.glob(mask_path+"*.png")

    inputs = []
    labels = []
    ids = []
    for om in output_masks:
        id = om.split('/')[-1][:-4]
        ids.append(id)
        inputs.append(generate_feature(id, image_path))
        labels.append(generate_label(id, mask_path))

    return ids, inputs, labels

def convert_to_vector(img_mask_filename):
    from scipy.misc import imread
    import numpy as np
    try:
        image = np.zeros(shape=(101,101))
        image = imread(img_mask_filename)
        image = image.flatten(order="F")/65535
        image.astype(np.float32)
        return image

    except Exception as e:
        import sys
        print("Couldn't load mask due to {}".format(e))
        sys.exit(-2)


def generate_label(id, mask_path):
    """ Generates label for training mask path and id
    Args:
        id: id of image
        mask_path: base path for masks

    Returns:
            A numpy vector which is of shape (101*101)
    """
    from scipy.misc import imread
    import numpy as np
    mask_image_filename = mask_path+id+".png"
    return convert_to_vector(mask_image_filename)
    

def generate_feature(id, image_path):
    """ Generates feature (input image) for training given image path and id
    Args:
        id: id of image
        image_path: base image path

    Returns:
            A numpy vector which is of shape (101, 101,3)
    """
    from scipy.misc import imread
    import numpy as np
    input_image_filename = image_path+id+".png"
    feature = np.zeros(shape=(101,101,3))
    try:
        feature = imread(input_image_filename)
        print("Parsed id {} ".format(str(id)))
        return feature
    except Exception as e:
        print("Could not load image due to {}".format(e))

def generate_test_features(test_image_path):
    import glob
    ids = []
    inputs = []
    image_paths = glob.glob(test_image_path+"*.png")
    for om in image_paths:
        id = om.split('/')[-1][:-4]
        ids.append(id)
        inputs.append(generate_feature(id, test_image_path))

    return ids, inputs
    
if __name__ == '__main__':
    from model.resnetModel import ResnetModel as Model
    from model.resnetModel import Config
    image_path = "/mnt/image_translation/data/train/images/"
    mask_path = "/mnt/image_translation/data/train/masks/"
    test_image_path = "/mnt/image_translation/data/test/images/"
    test_ids, test_inputs = generate_test_features(test_image_path)
    ids, inputs, labels = generate_features_and_labels(image_path, mask_path)
    config = Config()
    config.IMAGE_HEIGHT = 101
    config.IMAGE_WIDTH = 101

    import numpy as np
    inputs = np.asarray(inputs)
    labels = np.asarray(labels)
    import tensorflow as tf
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
            saver = tf.train.Saver()
            losses  = model.fit(sess, inputs, labels)
            saver.save(sess, '/results/models/toy_model')
            #saver.restore(sess, '/results/models/checkpoint')
            test_ids, labels  = model.predict(sess, np.asarray(test_inputs), ids =test_ids)
            import csv
            with open("/results/submission/results.csv", "w") as fp:
                csv_out = csv.writer(fp)
                csv_out.writerow(["id","rle_mask"])
                for id,label in zip(test_ids, labels):
                    csv_out.writerow((id,label))





