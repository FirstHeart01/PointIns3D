from spconv.pytorch.modules import SparseModule


class Model(SparseModule):
    """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """

    def __init__(self, in_channels, out_channels, train_cfg, norm_fn, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_cfg = train_cfg
        self.norm_fn = norm_fn
