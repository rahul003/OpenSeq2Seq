# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import range
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from open_seq2seq.utils.utils import deco_print
from .encoder_decoder import EncoderDecoderModel



class NoOpModel(EncoderDecoderModel):

  def _create_decoder(self):
    data_layer = self.get_data_layer()
    self.params['decoder_params']['tgt_vocab_size'] = (
        data_layer.params['tgt_vocab_size']
    )
    self.autoregressive = data_layer.params['autoregressive']
    if self.autoregressive:
      self.params['decoder_params']['GO_SYMBOL'] = data_layer.start_index
      self.params['decoder_params']['END_SYMBOL'] = data_layer.end_index
      self.tensor_to_chars = dense_tensor_to_chars
      self.tensor_to_char_params['startindex'] = data_layer.start_index
      self.tensor_to_char_params['endindex'] = data_layer.end_index

    return super(NoOpModel, self)._create_decoder()

  def maybe_print_logs(self, input_values, output_values, training_step):
    x, len_x = input_values['source_tensors']
    samples = output_values[0]
    x_sample = x[0]
    len_x_sample = len_x[0]
    y, len_y = input_values['target_tensors']
    y_sample = y[0]
    len_y_sample = len_y[0]
    
    print("src len {}\nsum={}\nshape={}".format(len_x, sum(len_x), x.shape))
    print("trg len {}\nsum={}\nshape={}".format(len_y, sum(len_y), y.shape))
    #print(x)
    #print(y)
    #print("bsz={}".format(np.sum(np.maximum(len_x, len_y) - 2)))
    #print(y_sample)
    #print(len_y_sample)
    #print(y_sample[:len_y_sample])
    import pickle
    with open("/tmp/mb_{}".format(training_step), "wb") as f:
        pickle.dump((x, len_x, y, len_y), f)
    return
    decoded_sequence = output_values
    y_one_sample = y[0]
    len_y_one_sample = len_y[0]
    decoded_sequence_one_batch = decoded_sequence[0]
    
    # we also clip the sample by the correct length
    true_text = "".join(map(
        self.get_data_layer().params['idx2char'].get,
        y_one_sample[:len_y_one_sample],
    ))
    pred_text = "".join(self.tensor_to_chars(
        decoded_sequence_one_batch,
        self.get_data_layer().params['idx2char'],
        **self.tensor_to_char_params
    )[0])
    sample_wer = levenshtein(true_text.split(), pred_text.split()) / \
        len(true_text.split())

    self.autoregressive = self.get_data_layer().params['autoregressive']
    self.plot_attention = False  # (output_values[1] != None).all()
    if self.plot_attention:
      attention_summary = plot_attention(
          output_values[1][0], pred_text, output_values[2][0], training_step)

    deco_print("Sample WER: {:.4f}".format(sample_wer), offset=4)
    deco_print("Sample target:     " + true_text, offset=4)
    deco_print("Sample prediction: " + pred_text, offset=4)

    if self.plot_attention:
      return {
          'Sample WER': sample_wer,
          'Attention Summary': attention_summary,
      }
    else:
      return {
          'Sample WER': sample_wer,
      }

  def finalize_evaluation(self, results_per_batch, training_step=None):
    total_word_lev = 0.0
    total_word_count = 0.0
    for word_lev, word_count in results_per_batch:
      total_word_lev += word_lev
      total_word_count += word_count

    total_wer = 1.0 * total_word_lev / total_word_count
    deco_print("Validation WER:  {:.4f}".format(total_wer), offset=4)
    return {
        "Eval WER": total_wer,
    }

  def evaluate(self, input_values, output_values):
    total_word_lev = 0.0
    total_word_count = 0.0

    decoded_sequence = output_values[0]
    decoded_texts = self.tensor_to_chars(
        decoded_sequence,
        self.get_data_layer().params['idx2char'],
        **self.tensor_to_char_params
    )
    batch_size = input_values['source_tensors'][0].shape[0]
    for sample_id in range(batch_size):
      # y is the third returned input value, thus input_values[2]
      # len_y is the fourth returned input value
      y = input_values['target_tensors'][0][sample_id]
      len_y = input_values['target_tensors'][1][sample_id]
      true_text = "".join(map(self.get_data_layer().params['idx2char'].get,
                              y[:len_y]))
      if self.get_data_layer().params['autoregressive']:
        true_text = true_text[:-4]
      pred_text = "".join(decoded_texts[sample_id])

      total_word_lev += levenshtein(true_text.split(), pred_text.split())
      total_word_count += len(true_text.split())

    return total_word_lev, total_word_count

  def infer(self, input_values, output_values):
    preds = []
    decoded_sequence = output_values[0]
    decoded_texts = self.tensor_to_chars(
        decoded_sequence,
        self.get_data_layer().params['idx2char'],
        **self.tensor_to_char_params
    )
    for decoded_text in decoded_texts:
      preds.append("".join(decoded_text))
    return preds, input_values['source_ids']

  def finalize_inference(self, results_per_batch, output_file):
    preds = []
    ids = []

    for result, idx in results_per_batch:
      preds.extend(result)
      ids.extend(idx)

    preds = np.array(preds)
    ids = np.hstack(ids)
    # restoring the correct order
    preds = preds[np.argsort(ids)]

    pd.DataFrame(
        {
            'wav_filename': self.get_data_layer().all_files,
            'predicted_transcript': preds,
        },
        columns=['wav_filename', 'predicted_transcript'],
    ).to_csv(output_file, index=False)

  def _get_num_objects_per_step(self, worker_id=0):
    """Returns number of audio frames in current batch."""
    data_layer = self.get_data_layer(worker_id)
    num_frames = tf.reduce_sum(data_layer.input_tensors['source_tensors'][1])
    return num_frames
