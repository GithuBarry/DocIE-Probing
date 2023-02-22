
local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: -1,
  max_span_width: 8,
  data_paths: {
    train: "data/scirex_test/train.json",
    validation: "data/scirex_test/dev.json",
    test: "data/scirex_test/test.json",
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
    num_epochs: 1,
    patience: 8,
    optimizer +: {
      lr: 5e-4
    }
  },
  target_task: "ner"
}

