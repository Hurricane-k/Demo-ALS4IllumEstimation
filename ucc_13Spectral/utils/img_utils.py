#%% python package import
import torch

#%% function 1
def get_uv_coord(hist_size, device='cpu', dtype=torch.float32, range=1.0):
    """ Gets uv-coordinate extra channels to augment each histogram as
    mentioned in the paper.

  Args:
    hist_size: histogram dimension (scalar).
    tensor: boolean flag for input torch tensor; default is true.
    normalize: boolean flag to normalize each coordinate channel; default
      is false.
    device: output tensor allocation ('cuda' or 'cpu'); default is 'cuda'.

  Returns:
    u_coord: extra channel of the u coordinate values; if tensor arg is True,
      the returned tensor will be in (1 x height x width) format; otherwise,
      it will be in (height x width) format.
    v_coord: extra channel of the v coordinate values. The format is the same
      as for u_coord.
  """

    u_coord, v_coord = torch.meshgrid(torch.arange(-(hist_size - 1) / 2,
                                                   ((hist_size - 1) / 2) + 1),
                                      torch.arange(-(hist_size - 1) / 2,
                                                   ((hist_size - 1) / 2) + 1),
                                      indexing='ij')  # uv could be negative
    scale = range / (hist_size - 1)
    u_coord.requires_grad = False
    v_coord.requires_grad = False
    u_coord = u_coord * scale
    v_coord = v_coord * scale
    u_coord = u_coord.to(device=device, dtype=dtype)
    v_coord = v_coord.to(device=device, dtype=dtype)
    return u_coord, v_coord

#%% function 2
def log_uv_to_rgb_torch(uv: torch.Tensor, channel_first=False):
    """ Converts log-chroma space to RGB.

    Args:
        uv: input color(s) in chroma log-chroma space.
        channel_first: boolean flag for input tensor format; default is false.

    Returns:
        color(s) in rgb space.
    """

    rb = torch.exp(-uv)
    if channel_first:
        r = rb[0]
        b = rb[1]
        rgb = torch.stack(
            [r, torch.ones_like(r, dtype=uv.dtype, device=uv.device), b],
            dim=0)
    else:
        r = rb[..., 0]
        b = rb[..., 1]
        rgb = torch.stack(
            [r, torch.ones_like(r, dtype=uv.dtype, device=uv.device), b],
            dim=-1)
    return rgb

#%% function 3
def rgb_to_log_uv_torch(rgb: torch.Tensor, channel_first=False):
    """ Converts RGB to log-chroma space.

        Args:
            rgb: input color(s) in rgb space.
            channel_first: boolean flag for input tensor format; default is false.

        Returns:
            color(s) in chroma log-chroma space.
        """

    log_rgb = torch.log(rgb + 1e-9)
    if channel_first:
        u = log_rgb[1] - log_rgb[0]
        v = log_rgb[1] - log_rgb[2]
        uv = torch.stack([u, v], dim=0)
    else:
        u = log_rgb[..., 1] - log_rgb[..., 0]
        v = log_rgb[..., 1] - log_rgb[..., 2]
        uv = torch.stack([u, v], dim=-1)
    return uv

#%% function 4
def compute_uv_histogram_torch(img: torch.Tensor,
                               bin_num=256,
                               boundary_value=2.0,
                               channel_first=False,
                               srgb=False):
    """
    Computes the uv histogram of the input image.

    Args:
        img: input image(s) in rgb space. Eithers in (height x width x 3) or
          (3 x height x width) format.
        bin_num: number of bins for the histogram.
        boundary_value: boundary value for the uv space.
        channel_first: boolean flag for input tensor format; default is false.
            False: spatial dimension x channel
            True: channel x spatial dimension

    Returns:
        uv histogram.
    """
    if channel_first:
        valid_mask = torch.all(img > 1e-9, dim=0)
        valid_img = img[:, valid_mask]
    else:
        valid_mask = torch.all(img > 1e-9, dim=-1)
        valid_img = img[valid_mask]
    valid_uv = rgb_to_log_uv_torch(valid_img, channel_first=channel_first)

    if channel_first:
        valid_mask = torch.all(valid_uv >= -boundary_value, dim=0) & torch.all(
            valid_uv <= boundary_value, dim=0)
        valid_uv = valid_uv[:, valid_mask]
        valid_img = valid_img[:, valid_mask]
        if srgb:
            illuminance = 0.2126 * valid_img[0] + 0.7152 * valid_img[
                1] + 0.0722 * valid_img[2]
        else:
            illuminance = torch.norm(valid_img, dim=0, p=2)
        valid_uv = valid_uv.permute(1, 0)
    else:
        valid_mask = torch.all(valid_uv >= -boundary_value,
                               dim=-1) & torch.all(valid_uv <= boundary_value,
                                                   dim=-1)
        valid_uv = valid_uv[valid_mask]
        valid_img = valid_img[valid_mask]
        if srgb:
            illuminance = 0.2126 * valid_img[..., 0] + 0.7152 * valid_img[
                ..., 1] + 0.0722 * valid_img[..., 2]
        else:
            illuminance = torch.norm(valid_img, dim=-1, p=2)

    hist, bins = torch.histogramdd(valid_uv,
                                   bins=[bin_num, bin_num],
                                   range=[-boundary_value, boundary_value] * 2,
                                   weight=illuminance)
    hist /= (hist.sum() + 1e-9)
    hist = torch.sqrt(hist)
    return hist

