from os import path
import nibabel as nib
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Volume:
    """
    Wrapper for nibabel.Nifti1Image, mainly for preprocessing (i.e. cropping)
    raw data and giving the option to save the cropped volume.
    """
    def __init__(self, filename: str):
        """
        Load NIFTI volume as a numpy array. The data values are coerced into the dimensions
        (217, 178, 60) and values are restricted to between [0, 10000].
        """
        self.filename = filename
        logger.debug('Loading %s', self.filename)
        self.image = nib.load(self.filename)
        data = self.image.get_fdata()

        self.slice_thickness = self.image.header.get_zooms()[self.image.header.get_dim_info()[2]]

        # Detect the bounding box of the foreground
        idx = np.nonzero(data > 0)
        x1, x2 = idx[0].min(), idx[0].max()
        y1, y2 = idx[1].min(), idx[1].max()
        z1, z2 = idx[2].min(), idx[2].max()

        self._cropped_affine = self.image.affine
        self._cropped_affine[:3, 3] = np.dot(self._cropped_affine, np.array([x1, y1, z1, 1]))[:3]

        # Crop the image
        data = data[x1:x2, y1:y2, z1:z2]
        self.cropped_data = data

        # Ivan converts the data to float32, but it's already float64
        # data = np.float32(data)

        # https://github.com/ilegorreta/Automatic-Fetal-Brain-Reconstruction-Pipeline/blob/4d56c05226eafd115d9ef6757aa57585200bd4e8/predict_resnet.py#L97-L98
        if data.shape > (217, 178, 60):
            logger.warning('%s exceeds dimensions (217, 178, 60)')

        data = np.nan_to_num(data)
        data[data < 0] = 0
        data[data >= 10000] = 10000
        data = np.expand_dims(data, axis=3)
        pad = np.zeros([217, 178, 60, 1], dtype=np.float32)
        pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data

        self.padded_data = pad

    def save_cropped(self, folder: str, name_suffix='_crop') -> str:
        dest = path.basename(self.filename)
        if name_suffix:
            nii = dest.index('.nii')
            dest = dest[:nii] + '_crop' + dest[nii:]
        dest = path.join(folder, dest)

        nim2 = self.image.__class__(self.cropped_data, self._cropped_affine, header=self.image.header)
        nib.save(nim2, dest)
        return dest
