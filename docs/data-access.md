# Data access and privacy

This project was developed with MIMIC-IV, a credentialed-access clinical dataset. MIMIC-IV data and patient-level derivatives are not distributed in this repository.

To reproduce the research, users must independently:

1. complete the required training and obtain access through PhysioNet;
2. accept the MIMIC-IV data-use agreement;
3. store the dataset outside the Git repository;
4. provide local paths through environment variables or configuration files.

DrugBank source files are also excluded because they are subject to separate licensing terms.

Do not commit raw or derived patient-level data, identifiers, prediction bundles, checkpoints, or licensed interaction records. The repository's `.gitignore` provides an additional safeguard, but users remain responsible for reviewing every commit.
