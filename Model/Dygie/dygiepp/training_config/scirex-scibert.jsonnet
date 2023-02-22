local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  max_span_width: 8,
  data_paths: {
    train: "data/scirex/train.json",
    validation: "data/scirex/dev.json",
    test: "data/scirex/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation:0.0,
    coref: 0.2,
    events: 0.0
  },
  data_loader +: {
    num_workers: 8
  },
  trainer +: {
    num_epochs: 8,
    patience: 2,
    optimizer +: {
      lr: 5e-4
    }
  },
  target_task: "ner"
}
