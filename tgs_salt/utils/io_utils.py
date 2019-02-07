import sys

def ids_to_ndarrays(ids, image_path_base, inputs=True,  args=None):
    """ Given a list of ids generate numpy ndarrays associated with those ids
        Args:
            ids : list of ids associated with generating images
            image_path_base: base where the image resides
            inputs : bool. If true, generate features. If false generate labels
            args: top level args
    """
    import numpy as np

    num_ids = len(ids)
    output = np.zeros(shape=(num_ids, 101, 101))

    for ix, id in enumerate(ids):
        if(inputs): 
            output[ix,:,:] = generate_feature(id, image_path_base)
        else:
            output[ix,:,:] = generate_label(id, image_path_base)

    if(inputs):
        return np.expand_dims(output, axis=-1)
    return output




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



def convert_to_vector(img_mask_filename):
    from PIL import Image
    import numpy as np
    image = np.zeros(shape=(101,101))
    try:
        image = np.array(Image.open(img_mask_filename).convert('L'))/255
        return image

    except Exception as e:
        import sys
        print("Couldn't load mask due to {}".format(e))
        sys.exit(-2) 


def generate_feature(id, image_path):
    """ Generates feature (input image) for training given image path and id
    Args:
        id: id of image
        image_path: base image path

    Returns:
            A numpy vector which is of shape (101, 101,3)
    """
    import numpy as np
    from PIL import Image
    input_image_filename = image_path+id+".png"
    feature = np.zeros(shape=(101,101))
    try:
        feature = np.array(Image.open(input_image_filename).convert('L'))/255
        print("Parsed id {} ".format(str(id)))
        return feature
    except Exception as e:
        print("Could not load image due to {}".format(e))
