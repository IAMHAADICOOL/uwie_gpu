import os
import cupy as cp
import cv2
import natsort
import xlwt
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

if __name__ == '__main__':
    folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
    path = os.path.join(folder, "InputImages")
    files = os.listdir(path)
    files = natsort.natsorted(files)

    for file in files:
        filepath = os.path.join(path, file)
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            print('******** Processing file:', file)
            
            # Read image using OpenCV (NumPy array)
            img = cv2.imread(filepath)

            # Convert image to CuPy array for GPU processing
            img = cp.asarray(img)

            # Apply histogram stretching
            img = stretching(img)

            # Convert back to NumPy for scene radiance processing
            img = cp.asnumpy(img)
            sceneRadiance = sceneRadianceRGB(img)

            # Apply HSV stretching
            sceneRadiance = HSVStretching(sceneRadiance)

            # Convert back to CuPy for final processing
            sceneRadiance = cp.asarray(sceneRadiance)

            # Convert back to NumPy for saving
            sceneRadiance = cp.asnumpy(sceneRadiance)
            cv2.imwrite(os.path.join('OutputImages', f'{prefix}_ICM.jpg'), sceneRadiance)
