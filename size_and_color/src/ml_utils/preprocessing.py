# Third Party Libraries
from monai.transforms import MapTransform
import torch


class HENormalization(MapTransform):
    """
    Apply H&E normalization to images.
    """
    def __init__(self, keys, normalizer, method):
        super().__init__(keys)
        self.normalizer = normalizer
        self.method = method

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if self.method == 'reinhard':
                img = self.normalizer.normalize(I=d[key])
            else:
                img, _, _ = self.normalizer.normalize(I=d[key])
            d[key] = img.permute(2, 0, 1)
        return d


class ToGrayscale(MapTransform):
    """
    Convert images to grayscale.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            img = img.float()
            img = torch.mean(img, dim=0, keepdim=True)
            d[key] = img
        return d