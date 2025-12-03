---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4000
- loss:CosineSimilarityLoss
widget:
- source_sentence: service now access for hello please advised has logged side hold
    until further notice please log let user has created thanks best regards analyst
    tuesday hi please his level kind regards application engineer surrey surrey please
    consider environment printing regarded confidential received error please notify
    destroy immediately statements intent shall become binding confirmed hard copy
    by signatory contents relate dealings other companies under control details which
    found
  sentences:
  - Access
  - HR Support
  - Miscellaneous
- source_sentence: notification draft ad sync tuesday draft ad sync hello please prepare
    questions please let hello everyone please informed starting wednesday active
    directory means information present gal details location grade details present
    important things more discrepancies details present gal information found same
    displayed continue leaver form process until leaver form details completed please
    follow further official communication encounter issues please hesitate kind regards
    thank holiday notice re draft ad sync thank sync scheduled evening thank re draft
    ad sync hello made change image displayed more visible added bullet kind regards
    tuesday re draft ad sync hi let proceed ad sync out better keep old structure
    until process moved think otherwise structure has various rights attached changing
    complicate things thank pm re draft ad sync hello regards mentioned discussed
    whole sync flow few times during weekly meeting decided postpone information about
    out date affecting blocking activities other among flows ticketing has approvals
    requests based structure out date blocking progress separate application other
    slightly bad implementing leaver process engaged delivering until re help completing
    leaver process wait transferred good wait ad part until then taking recommend
    ahead ad sync means other except receive date please advise approve ad sync users
    two options keep old structure until process moved previously discussed implement
    second script update back received confirmation possible needed thank holiday
    notice re draft ad sync good morning sync ad today latest received saw during
    weekly call decision postpone until leaver process transferred believe having
    sync two drastically affect leaver process during holidays high number leavers
    cross leaver process explaining process prepare communication draft once entire
    flow ready draft prepared sync believe out kind regards wednesday draft hello
    everyone please find draft ad sync believe out by please let believe additional
    details added hello everyone please informed starting information about date integrated
    active directory means information present gal details location grade details
    present important things more discrepancies details present gal information found
    same displayed continue leaver form process until leaver form details completed
    please follow further official communication encounter issues please hesitate
    kind regards officer
  sentences:
  - Purchase
  - Storage
  - Internal Project
- source_sentence: logging hours time oracle error sent tuesday november logging hours
    error hello currently unable enter hours receive error attached thank you kind
    regards monitoring engineer
  sentences:
  - Hardware
  - HR Support
  - Purchase
- source_sentence: windows forced upgrade query re forced upgrade hi want schedule
    upgrade friday november romanian possible please thank developer wednesday november
    pm forced upgrade importance high hello everyone upgraded workstation please disregard
    coming back announcement order inform upgrade mandatory possibility choose upgrade
    interval kindly asking give action possible estimated upgrade process complete
    approximately minutes please announcement attached file after proceed upgrade
    starting forced upgrade means upgrade process automatically workstation stop process
    assets upgraded date forced upgrade take notified least two times please understand
    responsibility backup files important note dependency remain version please notify
    possible workstation having less than free please article more information kb
    view article kb subscription store files please make files completely synchronized
    unlinking subscription please kb article more kb view article kb thank kind regards
    ext
  sentences:
  - Hardware
  - Purchase
  - Administrative rights
- source_sentence: reoccurring meeting to be cancelled pm pm meeting recurrent care
    si la pm meeting pm care
  sentences:
  - Hardware
  - Hardware
  - Access
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'reoccurring meeting to be cancelled pm pm meeting recurrent care si la pm meeting pm care',
    'Hardware',
    'Access',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9358, 0.0385],
#         [0.9358, 1.0000, 0.1271],
#         [0.0385, 0.1271, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                      | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                          | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 43.65 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 3.35 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.48</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1            | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-----------------|
  | <code>data roaming while working overseas tuesday december pm re roaming while working overseas thank much appreciate prompt helpful response kind regards planning officer december re roaming while working overseas hey quote agent makes things easier understand great calls roaming feature well traveler traveler also submitted cap removed until please make leave phone menu navigate mobile networks roaming option added screen capture help thank administrator phone tuesday december pm re roaming while working overseas hello off australia again departing thursday december returning work via work phone please per roaming cap removed certain tasks cannot hand over therefore respond promptly thanks kind regards planning officer re roaming while working overseas hi after contacting complete cap has removed added back onto number believe back please come these removed then please let thank administrator ext wednesday december pm roaming while working overseas importance high hello please advised queu...</code> | <code>Storage</code>  | <code>0.0</code> |
  | <code>po incomplete inactive requester tuesday october pm po incomplete inactive requester hello raised created po requisition active anymore po incomplete cannot process invoice close po best regards sp tar shared officer ext st</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | <code>Purchase</code> | <code>1.0</code> |
  | <code>additional screen not working screen with number en was working when arrived work images screen were coming up kindly resolved issue</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>Hardware</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 4.0   | 500  | 0.1335        |
| 8.0   | 1000 | 0.0436        |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->