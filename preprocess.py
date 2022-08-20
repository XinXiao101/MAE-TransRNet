import os
import SimpleITK as sitk
import scipy.ndimage as nd
import numpy as np
import scipy.ndimage as nd


def one_hot(img, C):
    out = np.zeros((C, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(C):
        out[i, ...] = img == i
    return out

def writeToNpy(path, target, training=False):

    for file in os.listdir(path):
        frames = []
        for f in os.listdir(os.path.join(path, file)):
            if f.find('patient') == -1:
                continue
            idx = f[16:18]
            if idx in frames:
                continue
            else:
                frames.append(idx)

        if len(frames) != 2:
            raise RuntimeError('why is not 2')

        if frames[0] < frames[1]:
            MinF = frames[0]
            MaxF = frames[1]
        else:
            MinF = frames[1]
            MaxF = frames[0]

        F1 = os.path.join(path, file, file + '_frame' + MinF)
        F2 = os.path.join(path, file, file + '_frame' + MaxF)

        xNmae = F1 + '.nii.gz'
        yNmae = F2 + '.nii.gz'
        xSegName = F1 + '_gt.nii.gz'
        ySegName = F2 + '_gt.nii.gz'

        x = sitk.GetArrayFromImage(sitk.ReadImage(xNmae))
        factor = [64 / x.shape[0], 128 / x.shape[1], 128 / x.shape[2]]
        newX = nd.zoom(x, factor, order=3)

        y = sitk.GetArrayFromImage(sitk.ReadImage(yNmae))
        newY = nd.zoom(y, factor, order=3)

        if training:
            xSeg = sitk.GetArrayFromImage(sitk.ReadImage(xSegName))
            newXseg = nd.zoom(xSeg, factor, order=0)
            newXseg = one_hot(newXseg, C=4)

            ySeg = sitk.GetArrayFromImage(sitk.ReadImage(ySegName))
            newYseg = nd.zoom(ySeg, factor, order=0)
            newYseg = one_hot(newYseg, C=4)
            np.savez(os.path.join(target, file), x=newX, y=newY, xSeg = newXseg, ySeg=newYseg)
        else:
            np.savez(os.path.join(target, file), x=newX, y=newY)
# pass

writeToNpy('./data/training', './npdata/training', training=False)
writeToNpy('./data/testing', './npdata/testing', training=True)
writeToNpy('./data/validation', './npdata/validation', training=True)