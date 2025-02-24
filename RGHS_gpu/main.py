import os
import cupy as cp
import cv2
import natsort
from LabStretching import LABStretching
from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

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

            # Read image and move to GPU
            img = cv2.imread(filepath)
            img = cp.asarray(img)

            # Apply color equalization
            img = RGB_equalisation(img)

            # Apply histogram stretching
            img = stretching(img)

            # Apply LAB stretching
            img = LABStretching(cp.asnumpy(img))  # Convert to NumPy for skimage processing

            # Convert back to CuPy for final processing
            img = cp.asarray(img)

            # Save the output image
            img = cp.asnumpy(img)
            cv2.imwrite(os.path.join(folder, 'OutputImages', f'{prefix}_RGHS.jpg'), img)
