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

from allennlp.pretrained import biaffine_parser_universal_dependencies_todzat_2017

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr_reader")
class SemanticScholarDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tagizer: Tokenizer = None,
                 tag_indexers: Dict[str, TokenIndexer] = None,
                 headizer: Tokenizer = None,
                 head_indexers: Dict[str, TokenIndexer] = None,
                 depizer: Tokenizer = None,
                 dep_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._tagizer = tagizer or WordTokenizer()
        self._tag_indexers = tag_indexers or {"tags": SingleIdTokenIndexer()}

        self._headizer = headizer or WordTokenizer()
        self._head_indexers = head_indexers or {"heads": SingleIdTokenIndexer()}

        self._depizer = depizer or WordTokenizer()
        self._dep_indexers = dep_indexers or {"deps": SingleIdTokenIndexer()}

        #self._ud_predictor = biaffine_parser_universal_dependencies_todzat_2017()
        #self._ud_predictor._model = self._ud_predictor._model.cuda()

        self.test_dict = {'words': ['At', 'least', 'one', 'dog', 'is', 'showing', 'its', 'tongue', '.'],
                          'pos': ['ADV', 'ADV', 'NUM', 'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'PUNCT'],
                          'predicted_dependencies': ['case', 'nmod', 'nummod', 'nsubj', 'root', 'ccomp', 'det', 'obj', 'punct'],
                          'predicted_heads': [2, 3, 4, 5, 0, 5, 8, 6, 5]}

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
        
                # Universal Dependency
                # arc_loss                (F) 
                # tag_loss                (F) 
                # loss                    (F) 
                # words                   (List of Words)
                # pos                     Size of words (List of part of speech) 
                # predicted_dependencies  Size of words (List of dependency)
                # predicted_heads         Size of words (List of Ints) Zero indexed
                # hierplane_tree          (Tree in nested map representation)
                ud_out = self.test_dict #self._ud_predictor.predict(tokens)

                tags = " ".join(ud_out['pos'])
                deps = " ".join(ud_out['predicted_dependencies'])

                heads = ud_out['predicted_heads']
                for i in range(len(heads)):
                  heads[i] = ud_out['words'][heads[i] - 1] if heads[i] != 0 else ud_out['words'][i]
                heads = " ".join(heads)
        
                if 'directory' in paper_json:
                    id = { 'identifier': paper_json['identifier'], 'directory': paper_json['directory'], 'sentence': paper_json['sentence'] }
                else:
                    id = { 'identifier': paper_json['identifier'], 'sentence': paper_json['sentence'] }
                yield self.text_to_instance(tokens, tags, heads, deps, id, label)

    @overrides
    def text_to_instance(self, tokens: str, tags: str, heads: str, deps: str, metadata: Dict[str, str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(tokenized_tokens, self._token_indexers)

        tokenized_tags = self._tagizer.tokenize(tags)
        tags_field = TextField(tokenized_tags, self._tag_indexers)

        tokenized_heads = self._headizer.tokenize(heads)
        heads_field = TextField(tokenized_heads, self._head_indexers)

        tokenized_deps = self._depizer.tokenize(deps)
        deps_field = TextField(tokenized_deps, self._dep_indexers)

        fields = {'tokens': tokens_field, 'tags': tags_field, 'heads': heads_field, 'deps': deps_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
