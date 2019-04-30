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
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "tag_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 10
        }
      }
    },
    "tag_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 10,
      "hidden_size": 20,
      "num_layers": 2,
      "dropout": 0.2
    },
    "head_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 10
        }
      }
    },
    "head_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 10,
      "hidden_size": 20,
      "num_layers": 2,
      "dropout": 0.2
    },
    "dep_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 10
        }
      }
    },
    "dep_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 10,
      "hidden_size": 20,
      "num_layers": 2,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 2248,
      "num_layers": 1,
      "hidden_dims": [2],
      "activations": ["linear"],
      "dropout": [0.0]
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 5,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    },
    "cuda_device": 0,
    "num_serialized_models_to_keep": 2
  }
}
