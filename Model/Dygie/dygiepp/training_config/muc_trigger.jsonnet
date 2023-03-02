
local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: 0,
  max_span_width: 8,
  data_paths: {
    train: "../../../Corpora/MUC/muc-trigger-v1/muc_dygie/muc_trigger/train.json",
    validation: "../../../Corpora/MUC/muc-trigger-v1/muc_dygie/muc_trigger/dev.json",
    test: "../../../Corpora/MUC/muc-trigger-v1/muc_dygie/muc_trigger/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation:0.0,
    coref: 0.2,
    events: 0.0
  },
  data_loader +: {
    num_workers: 0
  },
  trainer +: {
    num_epochs: 18,
    patience: 8,
    optimizer +: {
      lr: 5e-4
    }
  },
  target_task: "ner"
}

