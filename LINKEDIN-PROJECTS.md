# Portfolio upgrade and LinkedIn drafts

Original project numbers are retained. No LinkedIn post or website change has been published. Thesis, heart disease, Tulip/Sanad, telecom churn and diabetes are excluded from this upgrade.

## Access coverage

Local source copies, your linked public Kaggle German-credit notebook and relevant public benchmark datasets were inspected. Private Google Drive and Google Colab have **not** been accessed: the desktop Chrome-control tool fails before connecting. Consequently this does not claim to be a complete inventory of your cloud projects.

## 2 — Noor Desktop AI Assistant

Customized the open-source Jarvis assistant into Noor, a Windows desktop AI assistant with a redesigned interface, configurable cloud and local model providers, and text and voice interaction. My work includes provider setup and routing, live settings, Windows-protected credential storage, voice configuration, packaging and user documentation. Noor builds on ONEPUNCHMAN411/Jarvis; the original architecture and contributor credits are preserved.

**Skills:** Python, desktop application development, AI API integration, voice interfaces, Windows packaging.

**Positioning:** a product-engineering and open-source customization project. Already on your website; no website edit was made. Features are described from the existing project documentation, not newly revalidated end to end.

## 1 — Customer Satisfaction: PCA, Regularization and Boosting

Built a reproducible customer-satisfaction benchmark on 76,020 records, comparing regularization, PCA and gradient boosting. Used duplicate-aware partitions and training-only preprocessing to prevent leakage. The selected gradient-boosting model achieved held-out ROC-AUC 0.836 versus 0.779 for the same-split logistic baseline, with average precision improving from 0.142 to 0.194. Documented the limits of accuracy on a highly imbalanced target.

**Skills:** Python, scikit-learn, PCA, imbalanced classification, experiment design.

**Evidence and limitations:** The target is rare: a majority-only classifier already achieves about 96% accuracy. Report ROC-AUC and average precision, not accuracy as evidence of strong detection. At the default 0.5 threshold, minority recall remains very low; operational threshold/cost calibration is future work. The historical cross-validation score is not directly comparable to this new test split.

[Code, results and README](01-customer-satisfaction/README.md)

## 3 — Arabic Sentiment Classification with Word and Character Features

Developed an Arabic sentiment-classification pipeline with word and character TF-IDF features, duplicate checks and validation-based model selection. On a held-out set of 5,217 texts, the selected model achieved 70.85% accuracy and 0.709 macro F1, compared with 61.80% accuracy for an RBF-SVM baseline on the same split. Preserved negation and emoji and documented uncertainty in the source labels.

**Skills:** Arabic NLP, TF-IDF, logistic regression, text preprocessing, model evaluation.

**Evidence and limitations:** This is an internal benchmark on the supplied dataset; its label provenance is unverified. A random split is not proof of generalization to new authors, time periods, domains or dialects. The earlier notebook score used a different split and should not be treated as a controlled comparison.

[Code, results and README](03-arabic-sentiment/README.md)

## 7 — Cosmetics Catalog Analysis and Data Quality

Analyzed a 931-product cosmetics catalog using Python and pandas, exploring brand coverage, product types, pricing and ratings. Improved the analysis by preserving missing values and separating currencies. Identified substantial rating and currency gaps that prevent a reliable price–rating comparison, turning the project into a transparent, evidence-based catalog analysis rather than making unsupported market claims.

**Skills:** Python, pandas, exploratory data analysis, data quality, visualization.

**Evidence and limitations:** This catalog contains 931 products. Ratings are missing for 63.48% and currency for 60.47%; all 340 rated rows lack a currency. None of the rows with known currency has a rating, so a currency-controlled price–rating correlation cannot be estimated. Zero prices may represent unavailable prices. The privately saved Colab/Kaggle final version has not been inspected.

[Code, results and README](07-cosmetics/README.md)

## 8 — Technology Layoffs: Trends and Coverage

Built a reproducible analysis of 2,412 reported technology-layoff records, with monthly, industry and geographic views. Audited missing counts and reported observed totals separately from data coverage instead of filling missing events with invented numbers. Produced reusable tables and visualizations for a data-storytelling article, with clear limits on completeness and causal interpretation.

**Skills:** Python, pandas, time-series aggregation, missing-data analysis, visualization.

**Evidence and limitations:** The snapshot has 2,412 records from March 2020 to December 2025 and 372 missing layoff counts. Its observed total of 746,809 represents reported counts within this file, not a complete estimate of worldwide layoffs. Temporal and industry differences can reflect reporting coverage; they do not establish causes.

[Code, results and README](08-tech-layoffs/README.md)

## 9 — Fashion-MNIST Image Classification with a Regularized CNN

Developed a reproducible Fashion-MNIST image-classification experiment with a regularized CNN and an MLP baseline. Used separate training, validation and official test partitions, training-only normalization, and validation-selected checkpoints. Added learning curves, class-level error analysis and documented results. The selected model achieved 93.14% held-out accuracy and 0.931 macro F1.

**Skills:** PyTorch, computer vision, CNNs, regularization, experiment tracking.

**Evidence and limitations:** This is a small-image benchmark, not a validated classifier for real-world product photographs. MLP and CNN have different predefined training budgets. Tutorial notebooks were reference material; this refactor is a new reproducible experiment and does not claim a novel architecture.

[Code, results and README](09-fashion-mnist/README.md)

## 10 — Arabic Transformer Features for Sentiment Classification

Built an Arabic sentiment experiment using pretrained CAMeLBERT embeddings and a supervised classification head. Reused the duplicate-cleaned split from the classical NLP benchmark for a transparent comparison. Added masked pooling, validation-based regularization, CPU quantization and explicit documentation of the distinction between frozen-feature transfer learning and end-to-end fine-tuning. The selected model achieved 71.55% held-out accuracy and 0.715 macro F1. Accuracy was 0.71 percentage points above the classical TF-IDF model on the same test set; no statistically significant advantage is claimed.

**Skills:** Hugging Face Transformers, PyTorch, Arabic NLP, transfer learning, CPU inference.

**Evidence and limitations:** This is a frozen encoder plus trained linear head, not end-to-end transformer fine-tuning. Pretraining-corpus overlap and dataset label provenance are unverified. The classical and transformer results use the same task and partition but different text representations. A more complex model is not automatically a stronger result.

[Code, results and README](10-arabic-transformers/README.md)

## 16 — Weather Pattern Clustering with Temporal Validation

Refactored a weather-clustering workflow for 1.59 million minute-level records, using a 158,726-row sample. Added circular wind-direction features, temporal validation, training-only preprocessing and data-driven selection of cluster count. The chosen two-cluster solution achieved a held-out silhouette score of 0.325, with stability checks across random seeds. Documented the dataset mirror and limitations of interpreting unsupervised groups.

**Skills:** Unsupervised learning, MiniBatchKMeans, temporal validation, feature engineering, Python.

**Evidence and limitations:** Clustering has no labeled classification accuracy. The selected two-cluster solution has test silhouette 0.325 and cross-seed adjusted Rand agreement of 0.977 and 0.965. These summarize mathematical separation/stability, not meteorological ground truth or performance at other stations. Original notebook is collaborative coursework; preserve original collaborator credits if publishing the coursework.

[Code, results and README](16-weather-clustering/README.md)

## 17 — German Credit Risk: Evaluation and Preprocessing

Rebuilt a German-credit classification notebook as a reproducible pipeline with preprocessing contained inside cross-validation. Compared logistic regression, random forest and SVM and evaluated the accuracy-versus-minority-class tradeoff. The selected model achieved balanced accuracy of 0.742 on the held-out test set versus 0.710 for the baseline, while openly documenting the reduction in overall accuracy and the limits of this historical dataset.

**Skills:** scikit-learn pipelines, categorical encoding, cross-validation, imbalanced learning, model assessment.

**Evidence and limitations:** The selected balanced model increased test balanced accuracy from 0.710 to 0.742, while overall accuracy fell from 78.0% to 72.5%. This is a documented tradeoff, not an overall accuracy gain. Historical sensitive attributes are retained for assignment comparability; fairness and deployment suitability were not established. The older 71% result is not a controlled baseline.

[Code, results and README](17-german-credit/README.md)

## 18 — Breast-Cancer Diagnostic Benchmark with Scaled SVMs

Developed a reproducible classification benchmark on the Wisconsin Diagnostic Breast Cancer dataset. Used fold-local scaling and cross-validation to select an SVM, then achieved 97.37% accuracy on a held-out 114-record test set compared with 92.11% for an unscaled linear-SVM baseline. Included class-level evaluation, an accuracy confidence interval and clear limits on clinical interpretation.

**Skills:** Python, SVM, cross-validation, preprocessing pipelines, classification metrics.

**Evidence and limitations:** Selected test accuracy is 97.37% (111/114), with a Wilson 95% interval of approximately 92.55%–99.10%. The same-split unscaled baseline reached 92.11%. This is a small educational benchmark and has no clinical validation; it must not be represented as a diagnostic product.

[Code, results and README](18-breast-cancer/README.md)

## 14 — AI web-security extension

Yes, this was a Chrome Manifest V3 extension with a Flask backend. The collected source has wiring and error-handling gaps, including a DOM-based popup script incorrectly registered as a service worker and a missing icon. It remains a prototype; no scans or paid integrations were run. See [the extension audit and description](14-web-extension/README.md).

## 23 — NLP coursework: what you have

The collection includes text preprocessing; word, n-gram and character TF-IDF comparisons; average Word2Vec embeddings with logistic regression; an incomplete IMDB CNN assignment; Arabic GPT-2 generation; and multilingual sentiment/toxicity examples. A reference notebook saves 86.76% accuracy for word TF-IDF logistic regression, but credits an external tutorial, so that is not an independently verified original result to advertise. Best use: an attributed teaching article comparing text representations, or support material for projects 3 and 10. Complete unfinished cells and clearly identify adaptations before treating it as a standalone project.

## 24 — Neural networks from scratch: what you have

Small NumPy networks implement sigmoid/tanh activation, manual updates and toy predictions; one notebook also contains a basic Matplotlib chart. This demonstrates foundational understanding of neural networks rather than an application at scale. A strong learning article could explain forward propagation, gradients, learning rate and loss curves. A fuller future project should add finite-difference gradient checks and evaluate on a real held-out dataset. These are suggestions, not claims that those extensions were completed.

## 25 — YOLO, tracking and DepthAI: what you have

The files cover YOLOv8 detection, training, validation, export, tracking, segmentation, pose and classification examples, plus YOLOv7/DepthAI preparation. Saved evidence includes tiny tutorial training/export, a missing `yolo` command in one notebook and dataset-path failures in another. The collection is primarily tutorials and unfinished custom work. It is not yet evidence of a finished custom detector. A stronger future portfolio piece would select a real use case and licensed dataset, split by scene/video to avoid near-duplicate leakage, measure mAP50–95 and per-class precision/recall, measure inference latency, and show a short working demo with proper upstream attribution.
