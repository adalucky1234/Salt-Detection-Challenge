

import numpy as np

def image_to_output(img_np):
    """
    Takes an input mask in terms of a numpy array and convert it to the output format

    Args:
        Input numpy array representing the image

    Output:
        string
    """
    a = img_np.flatten(order='F')
    y= ''
    count = 0
    for i in range(len(a)):
        if(a[i] == 1):
            if(count == 0):
                y = y + ' ' +str(i+1)
            count +=1
        else:
            if(count != 0):
                y=y+ ' '+str(count)
                count = 0

    
    if(count!=0 and a[i] == 1):
        y = y + ' ' +str(count)
    
    return y[1:]


def mask_to_output(img_np):
    """
    Takes an input mask in terms of a numpy array and convert it to the output format

    Args:
        Input numpy array representing the image

    Output:
        string
    """
    x = img_np
    outputs = []
    for a in img_np:
        a = a.flatten(order='F')
        y= ''
        count = 0
        for i in range(len(a)):
            if(a[i] == 1):
                if(count == 0):
                    y = y + ' ' +str(i+1)
                count +=1
            else:
                if(count != 0):
                    y=y+ ' '+str(count)
                    count = 0

        
        if(count!=0 and a[i] == 1):
            y = y + ' ' +str(count)
        
        outputs.append(y[1:])
    return outputs

def test_all_images():
    import glob
    import pandas as pd
    from scipy.misc import imread
    input_masks = glob.glob("../data/train/masks/*.png")
    output = "../data/train.csv"
    
    for im in input_masks:
        try :
            id = im.split('/')[-1][:-4]
            output_df = pd.read_csv(output) 
            output_mask = output_df[output_df["id"]==id]["rle_mask"].values[0]

            from scipy.misc import imread
            np_img = (imread(im)/65535).astype(int)
            predicted = image_to_output(np_img)
            if(predicted != output_mask):
                print(id)
                print(predicted)
                print(output_mask)
        except Exception as e:
            print(str(e))
            print(id)


def IoU_one_mask(preds, true):
    """

    """
    preds =preds.astype(bool)
    true = true.astype(bool)
    print("Preds*true=>Intersection")
    print((preds*true).astype(int))
    print("Preds+true=>Union")
    print((preds+true).astype(int))
    Intersection = np.sum((preds*true))
    Union = np.sum((preds+true))

    return  Intersection.astype(float)/Union   



def IoU(preds, true):
    preds  =preds.astype(bool)
    true = true.astype(bool)

    assert(preds.shape == true.shape )  #preds.shape = (2,3,3), len(preds.shape) = 3
    num_axes = len(preds.shape)

    Intersection = (preds*true).astype(int)
    Union = (preds+true).astype(int)

    for i in range(num_axes-1):
        Intersection = np.sum(Intersection, axis=1)
        Union = np.sum(Union, axis=1)

    return np.divide(Intersection.astype(float), Union, out=np.ones(shape=Intersection.shape), where=Union!=0)


def GetEmptyMasks(masks):
    num_axes = len(masks.shape)
    sum = masks
    for i in range(num_axes-1):
        sum= np.sum(sum, axis=1)

    return sum == 0

def Precision(preds, true):
    #IoU_value = get_iou_vector(preds, true)

    empty_preds = GetEmptyMasks(preds)
    empty_true = GetEmptyMasks(true)
    #print(empty_true)
    #print(empty_preds)
    #TP = True_Postive(preds, true, threshold)
    #print(IoU_value)
    FP = np.sum(empty_true*np.invert(empty_preds))
    FN = np.sum(np.invert(empty_true)*empty_preds)
    raw_IoU = IoU(preds, true) 
    thresholds = np.arange(0.5, 1, 0.05)
    batch_size = int((preds.shape)[0])
    num_thresholds = int((thresholds.shape)[0])
    TP_mat = np.zeros((batch_size,batch_size)).astype(int)
    for i,t in enumerate(thresholds):
        TP_mat[:,i] = (raw_IoU>t).astype(int)




    TP = np.sum(TP_mat, axis=-1)
    IoUs = np.mean(TP_mat, axis=-1)
    Precision = np.divide(TP,(TP+FP+FN))
    return Precision, IoUs


def average_precision(preds, true):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = []

    for t in thresholds:
        precisions.append(Precision(true, preds, t))

    return np.mean(np.asarray(precisions))

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return metric

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    size = 3 
    num_masks = 15
    preds = np.random.randint(2, size=(num_masks, size, size))
    true = np.random.randint(2, size=(num_masks,size, size))
    true[0,:,:] = np.zeros(shape=(3,3))
    preds[1,:,:] = np.zeros(shape=(3,3))
    #preds = np.zeros(shape=(num_masks,size,size))
    #true = np.zeros(shape=(num_masks,size,size))
    print(true)
    print(preds)
    
    print(average_precision(preds, true))
    #print(Union(preds, true))
    #print(IoU(preds,true))
    

