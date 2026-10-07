# LinkedIn story drafts

Prepared drafts only; no posts have been published. Repository links are public and ready to include when posting. First-person wording is a draft for Nourah to review, not a record of a personal statement she has already made.

## Noor — shaping an AI assistant people can actually set up

An AI assistant needs more than a chat box. It needs a clear way to choose a provider, configure credentials, interact by voice and understand what the application can do.

That is the focus of Noor, my Windows desktop assistant customization based on ONEPUNCHMAN411’s open-source Jarvis project.

My work includes the Noor interface and branding, configurable cloud and local model providers, provider routing and setup, live settings, voice configuration, Windows-protected credential storage, packaging and user documentation.

The project gave me a practical setting for connecting model APIs to a desktop application and making the configuration understandable. Its existing documentation explains which components require separate credentials or downloads.

Credit matters: Noor builds on Jarvis’s architecture and tools. I preserve the upstream MIT license and creator attribution rather than claim the entire system from scratch.

Portfolio focus: desktop product engineering, AI API integration and open-source adaptation.

Suggested visual: an actual Noor interface screenshot from the existing project, with the provider setup or voice controls visible. A fresh end-to-end demo has not been recorded in this session.

---

## Arabic sentiment — the simpler model was surprisingly competitive

Would a pretrained Arabic transformer clearly outperform a carefully built classical model?

I compared two approaches on the same sentiment dataset: word-plus-character TF-IDF with logistic regression, and frozen CAMeLBERT representations with a trained classification head.

Before modeling, I removed duplicate normalized texts and conflicting labels, preserved negation and emoji, and created fixed training, validation and test partitions. Vocabulary, scaling and model selection stayed within the development data.

On 5,217 test examples, TF-IDF achieved 70.85% accuracy and the transformer achieved 71.55%. The classical approach also improved on an RBF-SVM baseline that reached 61.80% on the same split.

The transformer’s 0.71-point advantage was small. A paired comparison gave a 95% bootstrap interval from −0.63 to +2.03 percentage points and an exact paired-test p-value of 0.312. This experiment does not establish a reliable winner.

That is the useful story: a more complex representation deserves a measured comparison, and a strong simple baseline can remain competitive.

This is a benchmark on the supplied labels, whose provenance is still unverified. The transformer encoder was frozen; this was not end-to-end fine-tuning.

Code and figures: https://github.com/Nourah-Alotaibi/ai-data-science-portfolio
Suggested visual: 10-arabic-transformers/results/paired_comparison.png

---

## Customer satisfaction — why 96% accuracy needed a second look

A classifier could reach about 96% accuracy on this customer-satisfaction dataset simply by predicting the majority class. That would miss the people the model is supposed to help identify.

I rebuilt the experiment around that problem. Duplicate predictor rows stay together across partitions, preprocessing is fitted only on development data, and validation selects between regularization, PCA and gradient boosting.

The selected boosting model reached test ROC-AUC 0.836 versus 0.779 for the same-split logistic baseline. Average precision improved from 0.142 to 0.194.

But ranking cases well did not make the default 0.5 decision threshold useful: it detected only 2 of 602 dissatisfied examples.

In a follow-up, I selected an illustrative threshold using group-aware out-of-fold development predictions, maximizing minority-class F1 without using test labels. On the existing test set, recall changed from 0.33% to 45.02%, precision became 21.02%, and overall accuracy became 91.13%.

The tradeoff is visible: flagging more customers also creates more false positives. The threshold is not a validated business policy, and this follow-up reuses the reported test set rather than supplying fresh independent validation.

The story is about connecting metrics to the decision a model will support.

Code and figures: https://github.com/Nourah-Alotaibi/customer-satisfaction-pca-benchmark
Suggested visual: 01-customer-satisfaction/results/threshold_selection.png

---

## Fashion-MNIST — learning which clothing categories remain difficult

A shirt, coat and pullover can look similar in a tiny grayscale image. I used Fashion-MNIST to turn that challenge into a reproducible classification experiment.

The baseline was a dense neural network. The second model used convolutional layers, batch normalization, dropout, AdamW and a cosine learning-rate schedule.

I separated 48,000 training images from 12,000 validation images and selected checkpoints using validation accuracy. The official 10,000-image test set was reserved for the final comparison, and normalization statistics came from training pixels only.

The CNN reached 93.14% test accuracy; the MLP baseline reached 86.52%. Their predefined training budgets differed, so I do not describe this as a compute-matched comparison.

The confusion matrix tells the more useful story. Shirts remained difficult, while trousers, sandals and bags were much easier. Overall accuracy can hide those differences.

The result demonstrates an experimental workflow and error analysis on a standard benchmark. Generalization to real product photography still needs separate evaluation.

Code, learning curves and class-level results: https://github.com/Nourah-Alotaibi/ai-data-science-portfolio
Suggested visual: 09-fashion-mnist/results/test_confusion_matrix.png

---

## Cosmetics — the most useful finding was a limit in the data

I started with a cosmetics catalog and a natural question: can its prices and ratings support a useful comparison?

The catalog contained 931 products. I explored brand coverage, product types, prices, currencies and ratings, but first audited the fields that those comparisons depended on.

Ratings were missing for 63.48% of products and currency for 60.47%. More importantly, every rated row lacked a currency, while none of the rows with a known currency had a rating.

That means a reliable currency-controlled price–rating comparison is not supported by this file. A scatter plot that ignored the currency gaps could look convincing while telling an unreliable story.

The revised analysis preserves missing values, separates currencies and distinguishes product type from subcategory. NYX has the most entries in this catalog, but a catalog count is not a sales or market-share ranking.

For me, this project is a clear example of why exploratory analysis includes deciding which questions the available data can answer.

This story describes the executed local catalog analysis; my separately saved private cloud version has not been inspected in this session.

Code and figures: https://github.com/Nourah-Alotaibi/ai-data-science-portfolio
Suggested visual: 07-cosmetics/results/missingness.png

---

