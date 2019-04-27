{

  "train_data_path": "/home/jzda/code/nlvr/nlvr/nlvr2/data/smalltrain.json",
  "validation_data_path": "/home/jzda/code/nlvr/nlvr/nlvr2/data/smalltrain.json",

  "dataset_reader": {
    "type": "nlvr_reader",
    "token_indexers": {
       "tokens": {
         "type": "single_id",
         "lowercase_tokens": true
       }
    }
  },

  "model": {
    "type": "nlvr_classifier",
    "text_field_embedder": {
      "token_embedders": {
         "tokens": {
           "type": "embedding",
           "embedding_dim": 100,
           "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
           "trainable": true
         }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 16384,
      "num_layers": 4,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 92672,
      "num_layers": 11,
      "hidden_dims": [46336, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
      "activations": ["sigmoid", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"],
      "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 1
  },

  "trainer": {
    "num_epochs": 15,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
