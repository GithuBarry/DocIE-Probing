# MUC3/4 Dataset

MUC 3/4 dataset came from 1990s and is available
through [NIST](https://www-nlpir.nist.gov/related_projects/muc/muc_data/muc_data_index.html), is used throughout our
research. We used a small part of it for template filling. Many people has worked on this dataset.

## muc

MUC GTT is the version used by Xinya Du in his
work [Template Filling with Generative Transformers](https://github.com/xinyadu/gtt/) in 2020~2021

- Available for GTT model

- No triggers

- Certain templates, after removing some role, can be empty or become duplicates of each other

- Role filler indices of each role filler is measured in offsets of characters, not words.

- Role filler indicies of each role filler is merely the first occurrence of that word (merely the text.index(
  role_filler)), not the actual indices of the

- Role filler index is character-level

- Split into 200 test, 200 dev, 1300 training examples. That's why you might see "muc1700" I used around.

## muc-trigger

Wayne Chen worked on converting the above GTT-compatible dataset to [TANL](https://github.com/amazon-science/tanl)
-compatible. TANL works for the ACE dataset, and Wayne used this functionality of TANL to run on multi-template
prediction. His adaption of TANL and the converted dataset
is [available on his repo.](https://github.com/WayneChen2021/2022-spring-dov-level-ie/tree/main/TANL%20scripts/data/mucevent)

- Available for both GTT, TANL, and dygie++

- Contains triggers

- Role filler index is measured in offsets of words (as split by white space; sometimes `"` is a word too)

### muc-trigger-v0

The triggers Wayne used was done with Barry in Spring 2022, using a manual list of ranked trigger word per incident type. We both wrote our own script.

### muc-trigger-v1

Barry later updated triggers to the muc gtt dataset so triggers more adjacent to template role fillers are used when an example has multiple templates.
Script on upgrading the TANL triggers relies on `muc-trigger-v0` because the MUC-GTT to MUC-TANL was not available. 

## sentmuc

Sentence MUC is only splitting each document into sentences, each being a document with empty templates. Only used in obtaining embeddings for `SentCat` described in the paper (concatenated sentence embeddings as full text embeddings).
