# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import json
import time
import numpy as np
from collections import OrderedDict
incident_token_to_type = {'kidnapping': 'kidnapping', 'attack': 'attack', 'bombing': 'bombing', 'robbery': 'robbery', 'arson': 'arson', 'forced': 'forced work stoppage'} # for decoding

logger = logging.getLogger(__name__)
file_mode = None

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, docid, tokens, templates):
        """Constructs a InputExample.

        Args:
            docid: Unique id for the example.
            tokens: list. The tokens of the sequence.
            # extracts: of the format OrderedDict([('PerpInd', [[18, 18], [191, 191]]), ('PerpOrg', [[21, 29]]), ('Target', [[344, 347]]), ('Victim', []), ('Weapon', [[255, 255], [377, 377]])])
            templates: 
        """
        self.docid = docid
        self.tokens = tokens
        self.templates = templates


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, position_ids, label_ids, docid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.label_ids = label_ids
        self.docid = docid


def find_sub_list(m_tokens, doctext_tokens):
    m_len = len(m_tokens)
    for idx in (i for i, t in enumerate(doctext_tokens) if t == m_tokens[0]):
        if doctext_tokens[idx: idx + m_len] == m_tokens:
            return idx, idx + m_len - 1
    return -1, -1

def not_sub_string(candidate_str, entitys):
    for entity in entitys:
        mention_string = entity[0]
        if candidate_str in mention_string:
            return False
    return True

def read_golds_from_test_file(data_dir, tokenizer, debug=False):
    golds = OrderedDict()
    doctexts_tokens = OrderedDict()
    file_path = os.path.join(data_dir, "test.json")
    with open(file_path, encoding="utf-8") as f:
        example_cnt = 0
        for line in f:
            example_cnt += 1
            if example_cnt >3 and debug:
                break
            line = json.loads(line)
            if "#" not in line["docid"]:
                docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to int(0001)
            else:
                docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1].split("#")[0]) + (1+int(line["docid"].split("-")[-1].split("#")[1])) *100000
                # transform TST1-MUC3-0001#1 to 210001

            

            doctext, templates_raw = line["doctext"], line["templates"]

            templates = []
            for template_raw in templates_raw:
                template = OrderedDict()
                for role, value in template_raw.items():
                    if role == "incident_type":
                        template[role] = value
                    else:
                        template[role] = []
                        for entity_raw in value:
                            # first_mention_tokens = tokenizer.tokenize(entity[0][0])
                            # start, end = find_sub_list(first_mention_tokens, doctext_tokens)
                            # if start != -1 and end != -1:
                            #     template[role].append([start, end])
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(mention_offset_pair[0]) 
                            if entity:
                                template[role].append(entity)

                if template not in templates:
                    templates.append(template)

            # for role, entitys_raw in extracts_raw.items():
            #     extracts[role] = []
            #     for entity_raw in entitys_raw:
            #         entity = []
            #         for mention_offset_pair in entity_raw:
            #             entity.append(mention_offset_pair[0])
            #         if entity:
            #             extracts[role].append(entity)

            doctexts_tokens[docid] = tokenizer.tokenize(doctext)
            golds[docid] = templates
    # import ipdb; ipdb.set_trace()
    return doctexts_tokens, golds

def read_examples_from_file(data_dir, mode, tokenizer, debug=False):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if len(examples) >3 and debug:
                break
            line = json.loads(line)
            if mode == "train":
                docid = int(line["docid"].split("-")[-1]) # transform DEV-MUC3-0001 to 1
            else:
                
                first_label  = line["docid"].split("-")[0][-1]
                #docid = int(first_label if first_label.isdigit() else 5)*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to 10001
                print("docid", line["docid"])
                if "#" not in line["docid"]:
                    docid = int(first_label if first_label.isdigit() else 5)*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to 10001
                else:
                    docid = int(first_label if first_label.isdigit() else 5)*10000 + int(line["docid"].split("-")[-1].split("#")[0]) + (1+int(line["docid"].split("-")[-1].split("#")[1])) *100000
                    # transform TST1-MUC3-0001 to 010001

            doctext, templates_raw = line["doctext"], line["templates"]
            doctext_tokens = tokenizer.tokenize(doctext)

            # for role, entitys in extracts_raw.items():
            #     extracts[role] = []
            #     for entity in entitys:
            #         first_mention_tokens = tokenizer.tokenize(entity[0][0])
            #         start, end = find_sub_list(first_mention_tokens, doctext_tokens)
            #         if start != -1 and end != -1:
            #             extracts[role].append([start, end])

            templates = []
            example_entity_cnt = 0
            for template_raw in templates_raw:
                template = OrderedDict()
                for role, value in template_raw.items():
                    if role == "incident_type":
                        template[role] = value
                    else:
                        template[role] = []
                        for entity in value:
                            first_mention_tokens = tokenizer.tokenize(entity[0][0])
                            start, end = find_sub_list(first_mention_tokens, doctext_tokens)
                            if start != -1 and end != -1:
                                template[role].append([start, end])
                                example_entity_cnt += 1

                templates.append(template)
            examples.append(InputExample(docid=docid, tokens=doctext_tokens, templates=templates))

    return examples


def convert_examples_to_features(
    examples,
    # label_list,
    max_seq_length_src,
    max_seq_length_tgt,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    sep_template="[unused0]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """   

    features = []
    # max_num_entity_tgt = (max_seq_length_tgt - (1 + 5)) // 2 # excluding [CLS], [SEP] * 5
    max_num_entity_tgt = 15
    max_num_templates = 5
    saved_for_probing = dict()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        docid, tokens, templates = example.docid, example.tokens, example.templates
        # roles = sorted(extracts.keys())
        # trunkcating ``tokens'', special_tokens_count: account for [CLS] and [SEP]
        special_tokens_count = 3 + 7
        if len(tokens) > max_seq_length_src - special_tokens_count:
            tokens = tokens[: (max_seq_length_src - special_tokens_count)]

        src_tokens, tgt_tokens = [], []
        src_src_mask, src_tgt_mask, tgt_src_mask, tgt_tgt_mask = [], [], [], []
        src_segment_ids, tgt_segment_ids = [], []
        src_position_ids, tgt_position_ids = [], []
        label_ids = []
        role_to_src_token_offset = {}
        token_offset_to_src_token_offset = {}
        # incident_types = ['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'forced work stoppage', 'bombing / attack', 'attack / bombing']
        incident_types = ['kidnapping', 'attack', 'bombing', 'robbery', 'arson', 'bombing / attack', 'attack / bombing']
        incident_type_to_src_token_offset = {}

        ######### src_tokens, src_mask, src_segment_ids, src_position_ids 
        # [CLS]
        src_tokens.append(cls_token)

        # # roles
        # for role in roles:
        #     role_tokens = tokenizer.tokenize(role)
        #     src_tokens.append(role_tokens[0])
        #     role_to_src_token_offset[role] = len(src_tokens) - 1
        # # [SEP]
        # src_tokens.append(sep_token)

        # incident types
        for incident_t in incident_types:
            incident_t_token = tokenizer.tokenize(incident_t)[0]
            if incident_t_token not in incident_type_to_src_token_offset:
                src_tokens.append(incident_t_token)
                incident_type_to_src_token_offset[incident_t_token] = len(src_tokens) - 1
        # [SEP]_template
        src_tokens.append(sep_template)
        sep_template_offset = len(src_tokens) - 1

        # input tokens
        for idx, token in enumerate(tokens):
            src_tokens.append(token)
            token_offset_to_src_token_offset[idx] = len(src_tokens) - 1
        # [SEP]
        src_tokens.append(sep_token)
        src_segment_ids = [sequence_a_segment_id] * len(src_tokens)
        src_position_ids = list(range(len(src_tokens)))

        # convert to ids and padding
        src_tokens_ids = tokenizer.convert_tokens_to_ids(src_tokens)
        src_mask = [1 if mask_padding_with_zero else 0] * len(src_tokens_ids)

        padding_length = max_seq_length_src - len(src_tokens)
        src_tokens_ids += [pad_token] * padding_length
        src_mask += [0 if mask_padding_with_zero else 1] * padding_length
        src_segment_ids += [pad_token_segment_id] * padding_length
        src_position_ids += [0] * padding_length

        # import ipdb; ipdb.set_trace()

        ############ tgt_tokens, tgt_mask, tgt_segment_ids, tgt_position_ids, label_ids
        num_entity_span = 0
        # [CLS] (as start)
        tgt_tokens.append(cls_token)
        tgt_position_ids.append(0)
        # # each roles' spans
        # for role in roles:
        #     # role_tokens = tokenizer.tokenize(role)
        #     # tgt_tokens.append(role_tokens[0])
        #     for span in extracts[role]:
        #         if num_entity_span < max_num_entity_tgt and span[0] in range(len(tokens)) and span[1] in range(len(tokens)):
        #             num_entity_span += 1
        #             tgt_tokens.append(tokens[span[0]]) # span start token
        #             tgt_position_ids.a`ppend(token_offset_to_src_token_offset[span[0]])
        #             tgt_tokens.append(tokens[span[1]]) # span end token
        #             tgt_position_ids.append(token_offset_to_src_token_offset[span[1]])
        #     tgt_tokens.append(sep_token)
        #     tgt_position_ids.append(len(src_tokens) - 1) # to confirm
        # tgt_segment_ids = [1 - sequence_a_segment_id] * len(tgt_tokens)
        # label_ids = tgt_position_ids[1:]

        # each templates' spans
        num_templates = 0
        for template in templates:
            num_templates += 1
            if num_templates >  max_num_templates:
                break
            for role in template:
                if role == "incident_type":
                    incident_t_token = tokenizer.tokenize(template[role])[0]
                    if incident_t_token not in incident_type_to_src_token_offset:
                        break
                    tgt_tokens.append(incident_t_token)
                    tgt_position_ids.append(incident_type_to_src_token_offset[incident_t_token])
                else:
                    for span in template[role]:
                        if num_entity_span < max_num_entity_tgt and span[0] in range(len(tokens)) and span[1] in range(len(tokens)):
                            num_entity_span += 1
                            tgt_tokens.append(tokens[span[0]]) # span start token
                            tgt_position_ids.append(token_offset_to_src_token_offset[span[0]])
                            tgt_tokens.append(tokens[span[1]]) # span end token
                            tgt_position_ids.append(token_offset_to_src_token_offset[span[1]])
                tgt_tokens.append(sep_token)
                tgt_position_ids.append(len(src_tokens) - 1) # to confirm

            tgt_tokens.append(sep_template)
            tgt_position_ids.append(sep_template_offset)


        # [CLS] (as end)
        tgt_tokens.append(cls_token)
        tgt_position_ids.append(0)

        tgt_segment_ids = [1 - sequence_a_segment_id] * len(tgt_tokens)
        label_ids = tgt_position_ids[1:]

        # convert to ids and padding
        tgt_tokens_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
        tgt_mask = [1 if mask_padding_with_zero else 0] * len(tgt_tokens_ids)

        padding_length = max_seq_length_tgt - len(tgt_tokens)
        tgt_tokens_ids += [pad_token] * padding_length
        tgt_mask += [0 if mask_padding_with_zero else 1] * padding_length
        tgt_segment_ids += [pad_token_segment_id] * padding_length 
        tgt_position_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * (padding_length + 1)


        global file_mode
        doc_index = f"{file_mode}_{str(ex_index)}"
        saved_for_probing[doc_index] = dict()
        saved_for_probing[doc_index]["src_tokens"] = src_tokens
        saved_for_probing[doc_index]["tgt_tokens"] = tgt_tokens

        # import ipdb; ipdb.set_trace()

        ### get 2-d mask and get final input_ids, segment_ids, position_ids
        src_src_mask = np.array(src_mask)[None, :].repeat(max_seq_length_src, axis=0)
        tgt_src_mask = np.array(src_mask)[None, :].repeat(max_seq_length_tgt, axis=0)
        src_tgt_mask = np.full((max_seq_length_src, max_seq_length_tgt), 0 if mask_padding_with_zero else 1)
        seq_ids = np.array(list(range(len(tgt_tokens_ids))))
        try:
            tgt_tgt_causal_mask = seq_ids[None, :].repeat(max_seq_length_tgt, axis=0) <= seq_ids[:, None].repeat(max_seq_length_tgt, axis=1)
        except ValueError:
            import ipdb; ipdb.set_trace()
        tgt_tgt_mask = tgt_mask * tgt_tgt_causal_mask
        src_mask_2d = np.concatenate((src_src_mask, src_tgt_mask), axis=1)
        tgt_mask_2d = np.concatenate((tgt_src_mask, tgt_tgt_mask), axis=1)
        input_mask = np.concatenate((src_mask_2d, tgt_mask_2d), axis=0)

        input_ids = src_tokens_ids + tgt_tokens_ids
        segment_ids = src_segment_ids + tgt_segment_ids
        position_ids = src_position_ids + tgt_position_ids

        if len(input_ids) != 510:
            import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("docid: %d", docid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            # logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("position_ids: %s", " ".join([str(x) for x in position_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, position_ids=position_ids, label_ids=label_ids, docid=docid)
        )
        # import ipdb; ipdb.set_trace()
    json.dump(saved_for_probing, open(f"parsed_outputs{time.time():.0f}", "w+"))
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]