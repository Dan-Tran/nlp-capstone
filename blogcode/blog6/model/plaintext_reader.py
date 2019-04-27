from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr_reader")
class SemanticScholarDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                tokens = paper_json['sentence']
                label = paper_json['label']
                if 'directory' in paper_json:
                    id = { 'identifier': paper_json['identifier'], 'directory': paper_json['directory'] }
                else:
                    id = { 'identifier': paper_json['identifier'] }
                yield self.text_to_instance(tokens, id, label)

    @overrides
    def text_to_instance(self, tokens: str, metadata: Dict[str, str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(tokenized_tokens, self._token_indexers)

        fields = {'tokens': tokens_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
