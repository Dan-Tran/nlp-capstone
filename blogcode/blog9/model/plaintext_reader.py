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

# from allennlp.pretrained import biaffine_parser_universal_dependencies_todzat_2017

# from model.yolo import Yolo

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

        # self._ud_predictor = biaffine_parser_universal_dependencies_todzat_2017()
        # self._ud_predictor._model = self._ud_predictor._model.cuda()

        # self.yolo = Yolo()
        """
        self.test_dict = {'words': ['At', 'least', 'one', 'dog', 'is', 'showing', 'its', 'tongue', '.'],
                          'pos': ['ADV', 'ADV', 'NUM', 'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'PUNCT'],
                          'predicted_dependencies': ['case', 'nmod', 'nummod', 'nsubj', 'root', 'ccomp', 'det', 'obj', 'punct'],
                          'predicted_heads': [2, 3, 4, 5, 0, 5, 8, 6, 5]}
        """
        self._image_vecs = {}
        for line in open('/projects/instr/19sp/cse481n/DJ2/preprocessed/features_train_cuda.txt','r'):
            data = json.loads(line)
            self._image_vecs[data["identifier"]] = data
        for line in open('/projects/instr/19sp/cse481n/DJ2/preprocessed/features_dev_cuda_new.txt','r'):
            data = json.loads(line)
            self._image_vecs[data["identifier"]] = data

        with open('/projects/instr/19sp/cse481n/DJ2/parsing/trainparse.json', 'r') as trainfile:
            self.trainparse = json.load(trainfile)

        with open('/projects/instr/19sp/cse481n/DJ2/parsing/devparse.json', 'r') as devfile:
            self.devparse = json.load(devfile)

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
                
                # ud_out = self.test_dict
                ud_out = self.get_parse(tokens, 'directory' in paper_json)

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
        
                if 'directory' in paper_json:
                    id = { 'identifier': paper_json['identifier'], 'directory': paper_json['directory'], 'image_dict': self._image_vecs[paper_json['identifier']], 'sentence': paper_json['sentence'] }
                else:
                    id = { 'identifier': paper_json['identifier'], 'image_dict': self._image_vecs[paper_json['identifier']], 'sentence': paper_json['sentence'] }

                # TODO: Replace with pretrained outputs
                # lob = [(b'bottle', 0.9941088557243347, (271.6317443847656, 187.2192840576172, 63.3642463684082, 247.9810333251953)), (b'cup', 0.9783698916435242, (25.982481002807617, 227.9794158935547, 55.7009391784668, 86.13970184326172)), (b'bottle', 0.9579184055328369, (151.7715301513672, 174.62904357910156, 59.851802825927734, 222.2829132080078))]
                # rob = [(b'bottle', 0.9941088557243347, (271.6317443847656, 187.2192840576172, 63.3642463684082, 247.9810333251953)), (b'cup', 0.9783698916435242, (25.982481002807617, 227.9794158935547, 55.7009391784668, 86.13970184326172)), (b'bottle', 0.9579184055328369, (151.7715301513672, 174.62904357910156, 59.851802825927734, 222.2829132080078))]

                # Manual yolo detection w/o using preprocessing
                # lob = self.yolo.detect(self.get_left_link(id).encode())
                # rob = self.yolo.detect(self.get_right_link(id).encode())

                # Use yolo preprocessing
                lob = self.yolo_obj_detect(id, 0)
                rob = self.yolo_obj_detect(id, 1)

                if len(lob) == 0:
                    lob = [(b'null', 0.0, (0.0, 0.0, 0.0, 0.0))]
                if len(rob) == 0:
                    rob = [(b'null', 0.0, (0.0, 0.0, 0.0, 0.0))]

                lobstring = ' '.join(map(lambda x: x[0].decode('utf-8'), lob))
                lobinfo = list(map(lambda x: [x[1], x[2][0], x[2][1], x[2][2], x[2][3]], lob))
                id['left_object_info'] = lobinfo

                #print(lobstring)
                #print(lobinfo)

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
        # fields = {'tokens': tokens_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def get_parse(self, tokens, is_train):
        if is_train: # train image
            return self.trainparse[tokens]
        else: # dev image
            return self.devparse[tokens]

    def get_left_link(self, metadata: Dict[str, torch.LongTensor]) -> str:
        if 'directory' in metadata: # training image
            return "/projects/instr/19sp/cse481n/DJ2/images/train/" + str(metadata['directory']) + "/" + metadata['identifier'][:-2] + "-img0.png"
        else: # dev image
            return "/projects/instr/19sp/cse481n/DJ2/images/dev/" + metadata['identifier'][:-2] + "-img0.png"

    def get_right_link(self, metadata: Dict[str, torch.LongTensor]) -> str:
        if 'directory' in metadata: # training image
            return "/projects/instr/19sp/cse481n/DJ2/images/train/" + str(metadata['directory']) + "/" + metadata['identifier'][:-2] + "-img1.png"
        else: # dev image
            return "/projects/instr/19sp/cse481n/DJ2/images/dev/" + metadata['identifier'][:-2] + "-img1.png"

    # Side represents left image if 0, else right image
    def yolo_obj_detect(self, metadata: Dict[str, torch.LongTensor], side: int) -> list:
        preprocessing_path = "/projects/instr/19sp/cse481n/DJ2/nlp-capstone/blogcode/blog8/objdetection/" 
        image_path = ("train/" + str(metadata['directory']) + "/" + metadata['identifier'][:-2]) if 'directory' in metadata else ("dev/" + metadata['identifier'][:-2])
        side_indicator = "-img0.png" if side == 0 else "-img1.png"
                         
        full_path = preprocessing_path + image_path + side_indicator

        try:
          with open(full_path, 'rb') as f:
            return pickle.load(f)
        except:
          return []
