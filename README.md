# Prediction of phase-separation propensities of disordered proteins from sequence
Supporting data and code for:

*S. von Bülow, G. Tesei, F. K. Zaidi, T. Mittag, K. Lindorff-Larsen, Prediction of phase-separation propensities of disordered proteins from sequence. Proc. Natl. Acad. Sci. U.S.A. (2025).*

The phase-separation predictor can be run on Google Colab:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2024_buelow_PSpred/blob/main/PSLab.ipynb)

The CALVADOS simulation code is available [here](https://github.com/KULL-Centre/CALVADOS).

Simulation trajectories and density maps are deposited on the Electronic Research Data Archive (ERDA): [Link](doi.org/10.17894/ucph.e364274d-8094-4852-88ba-cb8985810a8d)

## List of content:
- **data**: Pandas dataframes with sequences, features and phase separation propensities for the training data, validation data and full IDRome. The folder also contains density maps for simulations excluded from the analysis. Also included is raw data for the structural analysis of the interface. 
- **data_revision**: Data added/modified during revision process.
- **models**: ML models.
- **scripts_colab**: Scripts supporting the Google Colab implementation.
- **PSLab.ipynb**: Google Colab.
- **example.fasta**: Fasta example file for Google Colab batch prediction.
- **figures.ipynb**: Jupyter notebook to reproduce plots for the paper figures.
- **figures_revision.ipynb**: Jupyter notebook to reproduce revised plots for the paper figures. The notebook requires CALVADOS, available [here](https://github.com/KULL-Centre/CALVADOS).
