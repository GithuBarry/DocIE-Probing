
local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-uncased",
  cuda_device: 0,
  max_span_width: 8,
  data_paths: {
    train: "../../../Corpora/MUC/muc-trigger-v1/sentmuc_dygie/muc_event_w_ner/train.json",
    validation: "../../../Corpora/MUC/muc-trigger-v1/sentmuc_dygie/muc_event_w_ner/dev.json",
    test: "../../../Corpora/MUC/muc-trigger-v1/sentmuc_dygie/muc_event_w_ner/test.json",
  },
  
  loss_weights: {
    ner: 0.0,
    relation:0.0,
    coref: 0.0,
    events: 1.0
  },
  data_loader +: {
    num_workers: 0
  },
  trainer +: {
      checkpointer: {
        num_serialized_models_to_keep: 16,
      },
    num_epochs: 16,
    patience: 16,
    optimizer +: {
      lr: 5e-4
    }
  },
  target_task: "events"
}

