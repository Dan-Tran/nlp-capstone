from typing import Dict
import json
import logging

import sys
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tagizer: Tokenizer = None,
                 tag_indexers: Dict[str, TokenIndexer] = None,
                 depizer: Tokenizer = None,
                 dep_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._tagizer = tagizer or WordTokenizer()
        self._tag_indexers = tag_indexers or {"tags": SingleIdTokenIndexer()}

        self._depizer = depizer or WordTokenizer()
        self._dep_indexers = dep_indexers or {"deps": SingleIdTokenIndexer()}

        # TODO: Preprocess test images
        self._image_vecs = {}
        for line in open('/projects/instr/19sp/cse481n/DJ2/preprocessed/features_test_cuda.txt','r'):
            data = json.loads(line)
            self._image_vecs[data["identifier"]] = data

        with open('/projects/instr/19sp/cse481n/DJ2/parsing/testparse.json', 'r') as testfile:
            self.testparse = json.load(testfile)

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
        
                ud_out = self.testparse[tokens]

                tags = " ".join(ud_out['pos'])
                deps = " ".join(ud_out['predicted_dependencies'])

                deps = deps.replace(':', '')

                heads = ud_out['predicted_heads'].copy()
                for i in range(len(heads)):
                  if int(heads[i]) != 0: 
                    heads[i] = ud_out['words'][int(heads[i]) - 1] 
                  else:
                    heads[i] = ud_out['words'][i]
                
                heads = " ".join(heads)
        
                id = { 'identifier': paper_json['identifier'], 'image_dict': self._image_vecs[paper_json['identifier']], 'sentence': paper_json['sentence'] }

                lob = self.yolo_obj_detect(id, 0)
                rob = self.yolo_obj_detect(id, 1)

                if len(lob) == 0:
                    lob = [(b'null', 0.0, (0.0, 0.0, 0.0, 0.0))]
                if len(rob) == 0:
                    rob = [(b'null', 0.0, (0.0, 0.0, 0.0, 0.0))]

                lobstring = ' '.join(map(lambda x: x[0].decode('utf-8'), lob))
                lobinfo = list(map(lambda x: [x[1], x[2][0], x[2][1], x[2][2], x[2][3]], lob))
                id['left_object_info'] = lobinfo

                robstring = ' '.join(map(lambda x: x[0].decode('utf-8'), rob))
                robinfo = list(map(lambda x: [x[1], x[2][0], x[2][1], x[2][2], x[2][3]], rob))
                id['right_object_info'] = robinfo

                yield self.text_to_instance(tokens, tags, heads, deps, lobstring, robstring, id, label)

    @overrides
    def text_to_instance(self, tokens: str, tags: str, heads: str, deps: str, lobstring: str, robstring: str, metadata: Dict[str, str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(tokenized_tokens, self._token_indexers)

        tokenized_tags = self._tagizer.tokenize(tags)
        tags_field = TextField(tokenized_tags, self._tag_indexers)

        tokenized_heads = self._tokenizer.tokenize(heads)
        heads_field = TextField(tokenized_heads, self._token_indexers)

        tokenized_deps = self._depizer.tokenize(deps)
        deps_field = TextField(tokenized_deps, self._dep_indexers)

        tokenized_lob = self._tokenizer.tokenize(lobstring)
        lob_field = TextField(tokenized_lob, self._token_indexers)

        tokenized_rob = self._tokenizer.tokenize(robstring)
        rob_field = TextField(tokenized_rob, self._token_indexers)

        fields = {'tokens': tokens_field, 'tags': tags_field, 'heads': heads_field, 'deps': deps_field, 'lob': lob_field, 'rob': rob_field, 'metadata': MetadataField(metadata)}

        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    # Side represents left image if 0, else right image
    def yolo_obj_detect(self, metadata: Dict[str, torch.LongTensor], side: int) -> list:
        preprocessing_path = "/projects/instr/19sp/cse481n/DJ2/nlp-capstone/blogcode/blog8/objdetection/" 
        image_path = "test/" + metadata['identifier'][:-2]
        side_indicator = "-img0.png" if side == 0 else "-img1.png"
                         
        full_path = preprocessing_path + image_path + side_indicator

        try:
          with open(full_path, 'rb') as f:
            return pickle.load(f)
        except:
          return []
