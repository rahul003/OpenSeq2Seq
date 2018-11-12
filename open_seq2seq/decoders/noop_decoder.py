# Copyright (c) 2018 NVIDIA Corporation
"""This module defines various fully-connected decoders (consisting of one
fully connected layer).

These classes are usually used for models that are not really
sequence-to-sequence and thus should be artificially split into encoder and
decoder by cutting, for example, on the last fully-connected layer.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os

import tensorflow as tf

from .decoder import Decoder


class NoOpDecoder(Decoder):
  """Simple decoder consisting of one fully-connected layer.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
    })

  def __init__(self, params, model,
               name="fully_connected_decoder", mode='train'):
    """Fully connected decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **output_dim** (int) --- output dimension.
    """
    super(NoOpDecoder, self).__init__(params, model, name, mode)


  def _decode(self, input_dict):
    """Creates TensorFlow graph for fully connected time decoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              'encoder_output': {
                "outputs": tensor with shape [batch_size, time length, hidden dim]
                "src_length": tensor with shape [batch_size]
              }
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'logits': logits with the shape=[time length, batch_size, tgt_vocab_size]
          'outputs': logits_to_outputs_func(logits, input_dict)
        }
    """
    print(input_dict)
    inputs = input_dict['encoder_output']['outputs']
    regularizer = self.params.get('regularizer', None)

    z = tf.reduce_sum(inputs)
    #print(inputs.get_shape().as_list())
    batch_size, _, n_hidden, _ = inputs.get_shape().as_list()
    # reshape from [B, T, A] --> [B*T, A].
    # Output shape: [n_steps * batch_size, n_hidden]
    #inputs = tf.reshape(inputs, [-1, n_hidden])

    '''
    logits = tf.reshape(
        logits,
        [batch_size, -1, self.params['tgt_vocab_size']],
        name="logits",
    )
    '''
    # converting to time_major=True shape
    #logits = tf.transpose(logits, [1, 0, 2])
    logits = tf.random_uniform(shape=(batch_size, batch_size), minval=0, maxval=100) + tf.cast(z, tf.float32)

    if 'logits_to_outputs_func' in self.params:
      outputs = self.params['logits_to_outputs_func'](logits, input_dict)
      return {
          'outputs': outputs,
          'logits': logits,
          'src_length': input_dict['encoder_output']['src_length'],
      }
    return {'logits': logits,
            'src_length': input_dict['encoder_output']['src_length']}

