# Prediction of phase separation propensities of disordered proteins from sequence
Supporting data and code for:

*von Bülow, S., Tesei, G., & Lindorff-Larsen, K. (2024). Prediction of phase separation propensities of disordered proteins from sequence. bioRxiv.*

The phase separation predictor can be run on Google Colaboratory:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2024_buelow_PSpred/blob/main/PSLab.ipynb)

## List of content:
- **calvados**: Scripts to run CALVADOS 2 simulations.
- **data**: Pandas dataframes with sequences, features and phase separation propensities for the training data, validation data and full IDRome. The folder also contains density maps as numpy arrays for those simulations that were not converged to a single dense phase and excluded from analysis.
- **models**: ML models.
- **scripts_colab**: Scripts supporting the Google Colab implementation.
- **PSLab.ipynb**: Google Colab.
- **example.fasta**: Fasta file example for Google Colab.
- **figures.ipynb**: Jupyter notebook to reproduce individual plots for the paper figures.
