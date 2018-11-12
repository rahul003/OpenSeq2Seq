# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models.noop_model import NoOpModel
from open_seq2seq.encoders.noop_encoder import NoOpEncoder
from open_seq2seq.decoders.noop_decoder import NoOpDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses.noop_loss import NoOpLoss
from open_seq2seq.optimizers.lr_policies import poly_decay


base_model = NoOpModel

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "num_gpus": 8,
  "batch_size_per_gpu": 32,

  "num_epochs": 50,
  "print_bench_info_steps" : 50,
  "print_loss_steps": 1,
  "profile_steps": 0,
  "profile_name": "ds2",
  "print_samples_steps": None,
  
  "save_checkpoint_steps": None,
  "save_summaries_steps": None,
  
  "eval_steps": 5000,

  "logdir": "experiments/librispeech",
  "bench_start": 100,
  
  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.001,
    "power": 0.5,
  },
  "larc_params": {
    "larc_eta": 0.001,
  },
  "dtype": "mixed",
  # weight decay
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": NoOpEncoder,
  "encoder_params": {
  },

  "decoder": NoOpDecoder,
  "decoder_params": {
  },

  "loss": NoOpLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "augmentation": {'time_stretch_ratio': 0.05,
                     'noise_level_min': -90,
                     'noise_level_max': -60},
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "~/librispeech/librivox-train-clean-100.csv",
      "~/librispeech/librivox-train-clean-360.csv",
      "~/librispeech/librivox-train-other-500.csv",
    ],
    "max_duration": 16.7,
    "shuffle": True,
    "synthetic": False,
    "synthetic_len": 500,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "~/data/librispeech/librivox-dev-clean.csv",
    ],
    "shuffle": False,
  },
}
