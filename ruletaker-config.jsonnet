local train_size = 70108;
local batch_size = 4;
local num_gradient_accumulation_steps = 4;
local num_epochs = 4;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-base";
local max_pieces = 384;
local dataset_dir = "/data/hzz5361/raw_data/rule-reasoning-dataset-V2020.2.5.0/original/depth-3/";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "rule_reasoning",
    "syntax": "rulebase",
    "pretrained_model": transformer_model,
    "max_pieces": max_pieces
  },
  "validation_dataset_reader": {
    "type": "rule_reasoning",
    "sample": -1,
    "pretrained_model": transformer_model,
    "max_pieces": max_pieces
  },
  "train_data_path": dataset_dir + "train.jsonl",
  "validation_data_path": dataset_dir + "dev.jsonl",
  "test_data_path": dataset_dir + "test.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "transformer_binary_qa",
    "num_labels": 2,
    "pretrained_model": transformer_model
  },
  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size
      }
  },
//   "iterator": {
//     "type": "basic",
//     "batch_size": batch_size
//   },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "betas": [0.9, 0.98],
      "weight_decay": weight_decay,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / (num_gradient_accumulation_steps * batch_size)),
    },
    // "validation_metric": "+accuracy",
    "validation_metric": "+EM",
    "checkpointer": {
        "num_serialized_models_to_keep": 1
    },
    // "should_log_learning_rate": true,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
    "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}