import numpy as np

def color_image(image, num_classes=4):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1], 21), dtype=np.uint8)
    palette = pascal_palette()

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        label = [0]*21
        label[i] = 1
        arr_2d[m] = label
    
    return arr_2d.astype(np.float32)

def pascal_classes():
    classes = {'aeroplane': 1,  'bicycle': 2,  'bird': 3,  'boat': 4,
               'bottle': 5,  'bus': 6,  'car': 7,  'cat': 8,
               'chair': 9,  'cow': 10, 'diningtable': 11, 'dog': 12,
               'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
               'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}
    
    return classes

def pascal_palette():
    palette = {(0,   0,   0): 0,
               (128,   0,   0): 1,
               (0, 128,   0): 2,
               (128, 128,   0): 3,
               (0,   0, 128): 4,
               (128,   0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64,   0,   0): 8,
               (192,   0,   0): 9,
               (64, 128,   0): 10,
               (192, 128,   0): 11,
               (64,   0, 128): 12,
               (192,   0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0,  64,   0): 16,
               (128,  64,   0): 17,
               (0, 192,   0): 18,
               (128, 192,   0): 19,
               (0,  64, 128): 20}
    
    return palette