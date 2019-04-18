from typing import Dict, Optional

import numpy
from overrides import overrides
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

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import torchvision.models as models

@Model.register("nlvr_classifier")
class SentimentClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != abstract_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

        # CNN Portion
        self.vgg16 = models.vgg16(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)


    def process_image(self, link: str) -> None:
        img_path = link
        img = load_img(img_path, target_size=(530, 700))
        img_data = img_to_array(img)
        img_data = numpy.expand_dims(img_data, axis=0)
        img_data = self.vgg16(img_data)

        x = self.conv1(img_data)
        x = self.conv2(x)
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.avg_pool2d(x, (19, 19))

        x = x.view(-1, self.num_flat_features(x))

        # print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_left_link(self, metadata: Dict[str, torch.LongTensor]) -> str:
        if 'directory' in metadata[0]: # training image
            return "/home/jzda/images/train/" + str(metadata[0]['directory']) + "/" + metadata[0]['identifier'][:-2] + "-img0.png"
        else: # dev image
            return "/home/jzda/images/dev/" + metadata[0]['identifier'][:-2] + "-img0.png"

    def get_right_link(self, metadata: Dict[str, torch.LongTensor]) -> str:
        if 'directory' in metadata[0]: # training image
            return "/home/jzda/images/train/" + str(metadata[0]['directory']) + "/" + metadata[0]['identifier'][:-2] + "-img1.png"
        else: # dev image
            return "/home/jzda/images/dev/" + metadata[0]['identifier'][:-2] + "-img1.png"

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                metadata: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pictures (CNN)
        left = self.get_left_link(metadata)
        left_image_vector = self.process_image(left)
        right = self.get_right_link(metadata)
        right_image_vector = self.process_image(right)
        # language (RNN)
        embedded_tokens = self.text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        encoded_tokens = self.abstract_encoder(embedded_tokens, tokens_mask)
        # combination + feedforward
        encoded_tokens_array = encoded_tokens.detach().numpy()
        concatenated_array = numpy.concatenate((left_image_vector[0], right_image_vector[0], encoded_tokens_array[0]), axis=None)
        concatenated_vector = torch.from_numpy(numpy.reshape(concatenated_array, (1, -1)))
        logits = self.classifier_feedforward(concatenated_vector.cpu())
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
