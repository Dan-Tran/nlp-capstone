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

        self.test_dict = {'arc_loss': 1.732508280838374e-05, 'tag_loss': 0.00041601393604651093, 'loss': 0.00043333900975994766, 'words': ['At', 'least', 'one', 'dog', 'is', 'showing', 'its', 'tongue', '.'
        ], 'pos': ['ADV', 'ADV', 'NUM', 'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'PUNCT'
        ], 'predicted_dependencies': ['case', 'nmod', 'nummod', 'nsubj', 'root', 'ccomp', 'det', 'obj', 'punct'
        ], 'predicted_heads': [
            2,
            3,
            4,
            5,
            0,
            5,
            8,
            6,
            5
        ], 'hierplane_tree': {'text': 'At least one dog is showing its tongue .', 'root': {'word': 'is', 'nodeType': 'root', 'attributes': ['VERB'
                ], 'link': 'root', 'spans': [
                    {'start': 17, 'end': 20
                    }
                ], 'children': [
                    {'word': 'dog', 'nodeType': 'nsubj', 'attributes': ['NOUN'
                        ], 'link': 'nsubj', 'spans': [
                            {'start': 13, 'end': 17
                            }
                        ], 'children': [
                            {'word': 'one', 'nodeType': 'nummod', 'attributes': ['NUM'
                                ], 'link': 'nummod', 'spans': [
                                    {'start': 9, 'end': 13
                                    }
                                ], 'children': [
                                    {'word': 'least', 'nodeType': 'nmod', 'attributes': ['ADV'
                                        ], 'link': 'nmod', 'spans': [
                                            {'start': 3, 'end': 9
                                            }
                                        ], 'children': [
                                            {'word': 'At', 'nodeType': 'case', 'attributes': ['ADV'
                                                ], 'link': 'case', 'spans': [
                                                    {'start': 0, 'end': 3
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {'word': 'showing', 'nodeType': 'ccomp', 'attributes': ['VERB'
                        ], 'link': 'ccomp', 'spans': [
                            {'start': 20, 'end': 28
                            }
                        ], 'children': [
                            {'word': 'tongue', 'nodeType': 'obj', 'attributes': ['NOUN'
                                ], 'link': 'obj', 'spans': [
                                    {'start': 32, 'end': 39
                                    }
                                ], 'children': [
                                    {'word': 'its', 'nodeType': 'det', 'attributes': ['DET'
                                        ], 'link': 'det', 'spans': [
                                            {'start': 28, 'end': 32
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {'word': '.', 'nodeType': 'punct', 'attributes': ['PUNCT'
                        ], 'link': 'punct', 'spans': [
                            {'start': 39, 'end': 41
                            }
                        ]
                    }
                ]
            }, 'nodeTypeToStyle': {'root': ['color5', 'strong'
                ], 'dep': ['color5', 'strong'
                ], 'nsubj': ['color1'
                ], 'nsubjpass': ['color1'
                ], 'csubj': ['color1'
                ], 'csubjpass': ['color1'
                ], 'pobj': ['color2'
                ], 'dobj': ['color2'
                ], 'iobj': ['color2'
                ], 'mark': ['color2'
                ], 'pcomp': ['color2'
                ], 'xcomp': ['color2'
                ], 'ccomp': ['color2'
                ], 'acomp': ['color2'
                ], 'aux': ['color3'
                ], 'cop': ['color3'
                ], 'det': ['color3'
                ], 'conj': ['color3'
                ], 'cc': ['color3'
                ], 'prep': ['color3'
                ], 'number': ['color3'
                ], 'possesive': ['color3'
                ], 'poss': ['color3'
                ], 'discourse': ['color3'
                ], 'expletive': ['color3'
                ], 'prt': ['color3'
                ], 'advcl': ['color3'
                ], 'mod': ['color4'
                ], 'amod': ['color4'
                ], 'tmod': ['color4'
                ], 'quantmod': ['color4'
                ], 'npadvmod': ['color4'
                ], 'infmod': ['color4'
                ], 'advmod': ['color4'
                ], 'appos': ['color4'
                ], 'nn': ['color4'
                ], 'neg': ['color0'
                ], 'punct': ['color0'
                ]
            }, 'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'
            }
        }
    }

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
