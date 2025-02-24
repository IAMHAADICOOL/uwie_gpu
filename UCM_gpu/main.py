import os
import cupy as cp
import cv2
import natsort
import xlwt
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

if __name__ == '__main__':
    starttime = datetime.datetime.now()

    folder = "/home/haadi/Single-Underwater-Image-Enhancement-and-Color-Restoration/images_iIT_MADRAS"
    path = os.path.join(folder, "InputImages")
    files = os.listdir(path)
    files = natsort.natsorted(files)

    for file in files:
        filepath = os.path.join(path, file)
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            print('********    file   ********', file)
            # Read image using cv2 (which returns a NumPy array)
            img = cv2.imread(filepath)
            # Process the image using the CuPy-enabled functions
            sceneRadiance = RGB_equalisation(img)
            sceneRadiance = stretching(sceneRadiance)
            sceneRadiance = HSVStretching(sceneRadiance)
            sceneRadiance = sceneRadianceRGB(sceneRadiance)
            # Write the output image
            cv2.imwrite(os.path.join(folder,'OutputImages', prefix + '_UCM_gpu.jpg'), sceneRadiance)

    endtime = datetime.datetime.now()
    elapsed = endtime - starttime
    print('time', elapsed)
