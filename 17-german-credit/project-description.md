# German Credit Risk: Evaluation and Preprocessing

Rebuilt a German-credit classification notebook as a reproducible pipeline with preprocessing contained inside cross-validation. Compared logistic regression, random forest and SVM and evaluated the accuracy-versus-minority-class tradeoff. The selected model achieved balanced accuracy of 0.742 on the held-out test set versus 0.710 for the baseline, while openly documenting the reduction in overall accuracy and the limits of this historical dataset.

**Skills:** scikit-learn pipelines, categorical encoding, cross-validation, imbalanced learning, model assessment.

**Supporting context:** The selected balanced model increased test balanced accuracy from 0.710 to 0.742, while overall accuracy fell from 78.0% to 72.5%. This is a documented tradeoff, not an overall accuracy gain. Historical sensitive attributes are retained for assignment comparability; fairness and deployment suitability were not established. The older 71% result is not a controlled baseline.
