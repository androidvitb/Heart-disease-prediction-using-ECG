---
library_name: transformers
license: apache-2.0
base_model: google/vit-large-patch32-384
tags:
- generated_from_trainer
model-index:
- name: sagarsahu_ECG-XRAY-ViT
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sagarsahu_ECG-XRAY-ViT

This model is a fine-tuned version of [google/vit-large-patch32-384](https://huggingface.co/google/vit-large-patch32-384) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0919

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.2773        | 1.0   | 116  | 0.1050          |
| 0.0739        | 2.0   | 232  | 0.0919          |


### Framework versions

- Transformers 4.48.1
- Pytorch 2.5.1
- Datasets 3.2.0
- Tokenizers 0.21.0
