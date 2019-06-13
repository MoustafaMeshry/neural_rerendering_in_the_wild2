import os.path as osp
import tensorflow as tf
import vgg16


vgg16_path = ''  # NOTE: set to the downloaded vgg16.npy


class PerceptualLoss:
  def __init__(self, x, y, image_shape, layers, w_layers, w_act=0.1):
    """
    Builds vgg16 network and computes the perceptual loss.
    """
    assert len(image_shape) == 3 and image_shape[-1] == 3
    assert osp.exists(vgg16_path), 'Cannot find %s'

    self.w_act = w_act
    self.vgg_layers = layers
    self.w_layers = w_layers
    batch_shape = [None] + image_shape  # [None, H, W, 3]

    vgg_net = vgg16.Vgg16(opts.vgg16_path)
    self.x_acts = vgg_net.get_vgg_activations(x, layers)
    self.y_acts = vgg_net.get_vgg_activations(y, layers)
    loss = 0
    for w, act1, act2 in zip(self.w_layers, self.x_acts, self.y_acts):
      loss += w * tf.reduce_mean(tf.square(self.w_act * (act1 - act2)))
    self.loss = loss

  def __call__(self):
    return self.loss

