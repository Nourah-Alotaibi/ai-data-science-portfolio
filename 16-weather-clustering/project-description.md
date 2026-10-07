# Weather Pattern Clustering with Temporal Validation

Refactored a weather-clustering workflow for 1.59 million minute-level records, using a 158,726-row sample. Added circular wind-direction features, temporal validation, training-only preprocessing and data-driven selection of cluster count. The chosen two-cluster solution achieved a held-out silhouette score of 0.325, with stability checks across random seeds. Documented the dataset mirror and limitations of interpreting unsupervised groups.

**Skills:** Unsupervised learning, MiniBatchKMeans, temporal validation, feature engineering, Python.

**Supporting context:** Clustering has no labeled classification accuracy. The selected two-cluster solution has test silhouette 0.325 and cross-seed adjusted Rand agreement of 0.977 and 0.965. These summarize mathematical separation/stability, not meteorological ground truth or performance at other stations. Original notebook is collaborative coursework; preserve original collaborator credits if publishing the coursework.
