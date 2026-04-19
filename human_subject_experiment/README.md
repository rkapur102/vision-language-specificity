# Human subject experiments

We conducted two human subject experiments, which are identical except for the exact task posed to the participants. `informativity_study/` contains the experiment participants completed when they were asked to choose the more informative description, and `preference_study/` contains the variant where participants were asked to simply choose the one they preferred (the main study in the paper).
In order to try out one of the study, simply clone this Github repository and open the `index.html` file (from either of the experiment folders) in your browser of choice. You will then be able to complete the experiment in the exact same way it was displayed to participants.

The `data/` folder contains the anonymized data from both experiments.

The `analysis/` folder contains an R Markdown file which has all human subject data analyses and statistical tests reported in the paper for replication. In particular, `analysis/analysis_paper.Rmd` includes the added human-subject covariate controls for average word frequency (excluding stopwords) and Flesch-Kincaid readability.
