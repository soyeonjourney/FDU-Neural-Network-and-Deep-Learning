import torch
import torchvision.transforms.functional as F


class Cutout(torch.nn.Module):
    """
    Apply cutout to the image.
    This operation applies a (2*half_size, 2*half_size) mask of zeros to a random location within image.
    The pixel values filled in will be of the value replace.
    """

    def __init__(self, p, half_size, replace=0):
        super().__init__()
        self.p = p
        self.half_size = int(half_size)
        self.replace = replace

    def forward(self, image):
        if torch.rand(1) < self.p:
            cutout_image = cutout(image, self.half_size, self.replace)
            return cutout_image
        else:
            return image

    def __repr__(self):
        return f"Cutout(p={self.p}, half_size={self.half_size})"


def cutout(img, half_size, replace):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    center_h, center_w = (
        torch.randint(high=h, size=(1,)),
        torch.randint(high=w, size=(1,)),
    )
    low_h, high_h = (
        torch.clamp(center_h - half_size, 0, h).item(),
        torch.clamp(center_h + half_size, 0, h).item(),
    )
    low_w, high_w = (
        torch.clamp(center_w - half_size, 0, w).item(),
        torch.clamp(center_w + half_size, 0, w).item(),
    )
    cutout_img = img.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    return F.to_pil_image(cutout_img)
