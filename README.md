# CFST Column Strength Prediction Web Application

A Streamlit web application for predicting Circular Concrete-Filled Steel Tube (CFST) column compressive strength and performing inverse design using CatBoost machine learning models.

## Research Paper

**"Enhancing CFST Compressive Strength Prediction and Inverse Design: Machine Learning Approach and Practical Implications"**

## Features

- **Forward Prediction**: Predict compressive strength (N) from structural parameters
- **Inverse Design**: Predict structural parameters given target compressive strength
  - Diameter D (mm)
  - Steel Tube Thickness t (mm)
  - Yield Strength of Steel fy (MPa)
  - Compressive Strength of Concrete fc (MPa)
  - Height L (mm)
  - Eccentricity of Load et (mm)
- **Data Visualization**: Statistical summaries, distributions, correlation heatmaps
- **Dataset Explorer**: Browse and filter the experimental dataset (1,287 samples)

## Live Demo

https://cfst-columns-ml-prediction.streamlit.app

## Technology Stack

- Streamlit
- CatBoost
- Pandas, NumPy
- Plotly

## Dataset

- 1,287 experimental samples
- 6 input features + 1 target variable
- Source: Published experimental data

## Model Performance

| Model | Target | Test R2 |
|-------|--------|---------|
| N | Compressive Strength | > 0.88 |
| D | Diameter | > 0.94 |
| t | Thickness | > 0.87 |
| fy | Yield Strength | > 0.84 |
| fc | Concrete Strength | > 0.76 |
| L | Height | > 0.81 |

## Contact

- khuongln@utt.edu.vn
- saeed.banihashemi@canberra.edu.au

## License

This application is part of academic research. Please cite the corresponding research paper when using this software.
