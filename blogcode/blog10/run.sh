rm -rf ser && allennlp train test.jsonnet -s ser --include-package model --include-package plaintext_reader
