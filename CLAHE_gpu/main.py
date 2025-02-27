import os
import cupy as cp
import cv2
import natsort
import xlwt
from skimage import exposure

from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE

if __name__ == '__main__':
    folder = "/home/haadi/Single-Underwater-Image-Enhancement-and-Color-Restoration/images_iIT_MADRAS"
    path = os.path.join(folder, "InputImages")
    files = os.listdir(path)
    files = natsort.natsorted(files)

    for file in files:
        filepath = os.path.join(path, file)
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            print('******** Processing file:', file)
            
            # Read image
            img = cv2.imread(filepath)
            
            # Convert image to CuPy array for faster processing
            img = cp.asarray(img)

            # Apply CLAHE enhancement
            sceneRadiance = RecoverCLAHE(img)

            # Convert back to NumPy and save the output image
            sceneRadiance = cp.asnumpy(sceneRadiance)
            cv2.imwrite(os.path.join(folder + 'OutputImages', f'{prefix}_CLAHE_gpu.jpg'), sceneRadiance)
