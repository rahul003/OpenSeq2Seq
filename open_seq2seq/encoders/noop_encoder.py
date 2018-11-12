# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from .encoder import Encoder

class NoOpEncoder(Encoder):
  """DeepSpeech-2 like encoder."""
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'row_conv_width': int,
        'data_format': ['channels_first', 'channels_last'],
        'bn_momentum': float,
        'bn_epsilon': float,
    })

  def __init__(self, params, model, name="ds2_encoder", mode='train'):
    super(NoOpEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """
    Returns:
      dict: dictionary with the following tensors::

        {
          'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
          'src_length': tensor, shape=[batch_size]
        }
    """

    print(input_dict)
    source_sequence, src_length = input_dict['source_tensors']
    input_layer = tf.expand_dims(source_sequence, axis=-1)
    batch_size = input_layer.get_shape().as_list()[0]
    input_layer += 1.0

    # Output shape: [batch_size, n_steps, n_hidden]
    return {
        'outputs': input_layer,
        'src_length': src_length,
    }
