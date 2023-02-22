local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: 0,
  max_span_width: 10,
  data_paths: {
    train: "data/promed/train.json",
    validation: "data/promed/dev.json",
    test: "data/promed/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation:0.0,
    coref: 0.2,
    events: 0.0
  },
  data_loader +: {
    num_workers: 16
  },
  trainer +: {
    num_epochs: 18,
    optimizer +: {
      lr: 5e-4
    }
  },
  target_task: "ner"
}
