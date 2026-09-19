# Safety-Aware Ensemble GNNs for Medication Recommendation

This research project investigates medication recommendation from longitudinal electronic health records using graph neural networks and safety-aware ensemble learning.

## Overview

The task is formulated as admission-level multi-label prediction of medication classes from longitudinal patient histories. The study uses MIMIC-IV and compares three graph neural network families:

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Relational Graph Convolutional Networks (R-GCN)

The models integrate diagnoses, procedures, medication history, laboratory measurements, and demographic and admission information. Patient histories are represented using temporally directed graphs to prevent future information from entering earlier admissions.

## Research focus

The project explores:

- modality-specific representation learning;
- adaptive feature gating;
- temporal graph construction;
- patient-disjoint evaluation;
- ensemble learning across complementary GNN architectures;
- drug–drug interaction-aware training and evaluation;
- explainability for clinical decision-support research.

The ensemble methodology is the principal contribution of the ongoing research. Implementation details will be released following publication.

## Data and reproducibility

MIMIC-IV is a credentialed-access clinical dataset and is not included in this repository. No patient-level data, identifiers, model prediction bundles, or licensed DrugBank files are distributed here.

Public code and documentation are being prepared to support reproducibility while respecting the data-use agreements for MIMIC-IV and DrugBank.

## Tools

Python · PyTorch · PyTorch Geometric · scikit-learn · pandas · NumPy · SQL

## Status

Active research project. The repository is being prepared for public release alongside the thesis and related publication.
