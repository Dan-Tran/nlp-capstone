{

  "train_data_path": "/projects/instr/19sp/cse481n/DJ2/nlvr/nlvr2/data/train.json",
  "validation_data_path": "/projects/instr/19sp/cse481n/DJ2/nlvr/nlvr2/data/dev.json",

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
    "type": "nlvr_test_classifier",
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
    "tag_embedder": {
      "type": "basic",
      "token_embedders": {
        "tags": {
          "type": "embedding",
          "embedding_dim": 50
        }
      }
    },
    "head_embedder": {
      "token_embedders": {
         "tokens": {
           "type": "embedding",
           "embedding_dim": 100,
           "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
           "trainable": true
         }
      }
    },
    "dep_embedder": {
      "type": "basic",
      "token_embedders": {
        "deps": {
          "type": "embedding",
          "embedding_dim": 50
        }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 300,
      "hidden_size": 500,
      "num_layers": 2,
      "dropout": 0.2
    },
    "ob_embedder": {
      "token_embedders": {
         "tokens": {
           "type": "embedding",
           "embedding_dim": 100,
           "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
           "trainable": true
         }
      }
    },
    "ob_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 105,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 2424,
      "num_layers": 4,
      "hidden_dims": [16, 8, 4, 2],
      "activations": "linear",
      "dropout": 0.0
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 40,
    "patience": 7,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    },
    "cuda_device": 0,
    "num_serialized_models_to_keep": 2
  }
}
