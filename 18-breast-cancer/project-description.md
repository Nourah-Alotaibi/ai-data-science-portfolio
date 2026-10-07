# Breast-Cancer Diagnostic Benchmark with Scaled SVMs

Developed a reproducible classification benchmark on the Wisconsin Diagnostic Breast Cancer dataset. Used fold-local scaling and cross-validation to select an SVM, then achieved 97.37% accuracy on a held-out 114-record test set compared with 92.11% for an unscaled linear-SVM baseline. Included class-level evaluation, an accuracy confidence interval and clear limits on clinical interpretation.

**Skills:** Python, SVM, cross-validation, preprocessing pipelines, classification metrics.

**Supporting context:** Selected test accuracy is 97.37% (111/114), with a Wilson 95% interval of approximately 92.55%–99.10%. The same-split unscaled baseline reached 92.11%. This is a small educational benchmark and has no clinical validation; it must not be represented as a diagnostic product.
