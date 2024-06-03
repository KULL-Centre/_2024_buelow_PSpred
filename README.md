# Prediction of phase separation propensities of disordered proteins from sequence
Supporting data and code for:

*von Bülow, S., Tesei, G., & Lindorff-Larsen, K. (2024). Prediction of phase separation propensities of disordered proteins from sequence. bioRxiv.*

The phase separation predictor can be run on Google Colab:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2024_buelow_PSpred/blob/main/PSLab.ipynb)

Simulation trajectories and density maps are deposited on the Electronic Research Data Archive (ERDA): [Link](https://sid.erda.dk/sharelink/hlZfnFz4AM)

## List of content:
- **calvados**: Scripts to run CALVADOS 2 simulations.
- **data**: Pandas dataframes with sequences, features and phase separation propensities for the training data, validation data and full IDRome. The folder also contains density maps for simulations excluded from the analysis. Also included is raw data for the structural analysis of the interface. 
- **models**: ML models.
- **scripts_colab**: Scripts supporting the Google Colab implementation.
- **PSLab.ipynb**: Google Colab.
- **example.fasta**: Fasta example file for Google Colab batch prediction.
- **figures.ipynb**: Jupyter notebook to reproduce plots for the paper figures.
