from typing import Dict, Optional

import numpy
from overrides import overrides
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from keras.preprocessing.image import img_to_array, load_img

import json
import torchvision.models as models

from model.yolo import Yolo

# outfile = open("preprocess.txt", "w")

@Model.register("nlvr_test_classifier")
class SentimentClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 abstract_encoder: Seq2VecEncoder,
                 ob_embedder: TextFieldEmbedder,
                 ob_encoder: Seq2VecEncoder,
                 tag_embedder: TextFieldEmbedder,
                 head_embedder: TextFieldEmbedder,
                 dep_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward

        self.tag_embedder = tag_embedder
        self.head_embedder = head_embedder
        self.dep_embedder = dep_embedder

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)

        self.ob_embedder = ob_embedder
        self.ob_encoder = ob_encoder


    def process_image(self, metadata: Dict[str, torch.LongTensor], focus: str) -> None:
        """
            img = map(lambda x: load_img(x, target_size=(200, 200)), link)
            img_data = torch.tensor(list(map(img_to_array, img))).permute(0, 3, 1, 2).cuda()

            x = F.max_pool2d(self.conv1(img_data), (4, 4))
            x = F.max_pool2d(self.conv2(x), (2, 2))
            x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

            x = x.view(-1, self.num_flat_features(x))

            #print(x.shape)
        """
        image_tensor = torch.zeros(len(metadata), 512).cuda()
        for i, item in enumerate(metadata):
            if "left" in focus:
                left_vec = torch.FloatTensor(metadata[i]["image_dict"]["left_vec"]).unsqueeze(0).cuda()
                image_tensor.index_copy_(0, torch.tensor([i]).cuda(), left_vec)
            else:
                right_vec = torch.FloatTensor(metadata[i]["image_dict"]["right_vec"]).unsqueeze(0).cuda()
                image_tensor.index_copy_(0, torch.tensor([i]).cuda(), right_vec)

        return image_tensor

    # Temporary, we should pretrain this
    #def detect_objects(self, link: str) -> None:

    #    reslist = list(map(lambda x: self.yolo.detect(x.encode()), link))
    #    print(reslist)

    #    # Fail to get detection output TEMPORARY
    #    raise RuntimeError

    #    return reslist

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: Dict[str, torch.LongTensor],
                heads: Dict[str, torch.LongTensor],
                deps: Dict[str, torch.LongTensor],
                lob: Dict[str, torch.LongTensor],
                rob: Dict[str, torch.LongTensor],
                metadata: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        # pictures (CNN)
        left_image_encoding = self.process_image(metadata, "left")
        right_image_encoding = self.process_image(metadata, "right")

        # language (RNN)
        embedded_tokens = self.text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)

        # Universal Dependencies
        embedded_tags = self.tag_embedder(tags)
        embedded_heads = self.head_embedder(heads)
        embedded_deps = self.dep_embedder(deps)

        # Objects
        embedded_lob = self.ob_embedder(lob)
        lob_mask = util.get_text_field_mask(lob)
        lobinfo = torch.tensor(list(map(lambda x: x['left_object_info'], metadata))).cuda()

        embedded_rob = self.ob_embedder(rob)
        rob_mask = util.get_text_field_mask(rob)
        robinfo = torch.tensor(list(map(lambda x: x['right_object_info'], metadata))).cuda()

        # Object Encoding
        cat_lob = torch.cat((embedded_lob, lobinfo), dim=2)
        encoded_lob = self.ob_encoder(cat_lob, lob_mask)

        cat_rob = torch.cat((embedded_rob, robinfo), dim=2)
        encoded_rob = self.ob_encoder(cat_rob, rob_mask)

        #print(embedded_tokens.shape)
        #print(embedded_tags.shape)
        #print(embedded_heads.shape)
        #print(embedded_deps.shape)


        # Concatentation
        # Somehow concatenate so that each element is [token tag head dep] in the sequence
        concatenated_embedding = torch.cat((embedded_tokens, embedded_tags, embedded_heads, embedded_deps), dim=2)

        # RNN Encoding
        encoded_tokens = self.abstract_encoder(concatenated_embedding, tokens_mask)

        # combination + feedforward
        concatenated_encoding = torch.cat((left_image_encoding, right_image_encoding, encoded_tokens, encoded_lob, encoded_rob), dim=1)
        logits = self.classifier_feedforward(concatenated_encoding)
        output_dict = {'logits': logits}

        # result = F.softmax(logits) # debug
        # max = torch.argmax(result, dim=1) # debug
        # print(max)
        # print(tokens)

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1) # softmax over the rows, dim=0 softmaxes over the columns
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
