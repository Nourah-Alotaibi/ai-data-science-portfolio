# Arabic Transformer Features for Sentiment Classification

Built an Arabic sentiment experiment using pretrained CAMeLBERT embeddings and a supervised classification head. Reused the duplicate-cleaned split from the classical NLP benchmark for a transparent comparison. Added masked pooling, validation-based regularization, CPU quantization and explicit documentation of the distinction between frozen-feature transfer learning and end-to-end fine-tuning. The selected model achieved 71.55% held-out accuracy and 0.715 macro F1. Accuracy was 0.71 percentage points above the classical TF-IDF model on the same test set; no statistically significant advantage is claimed.

**Skills:** Hugging Face Transformers, PyTorch, Arabic NLP, transfer learning, CPU inference.

**Supporting context:** This is a frozen encoder plus trained linear head, not end-to-end transformer fine-tuning. Pretraining-corpus overlap and dataset label provenance are unverified. The classical and transformer results use the same task and partition but different text representations. A more complex model is not automatically a stronger result.
