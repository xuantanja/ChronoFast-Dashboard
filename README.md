# Chronofast-Dashboard for Exploring Fasting States


This project was implemented as part of the master`s thesis at the Hasso Plattner Institute. The goal of the project is to build a prototype that classifies the fasting state of input data. Training data for the machine and time series classification models was the ChronoFast study conducted by Deutsches Institut für Ernährungsforschung Potsdam-Rehbrücke (DIfE).


## Structure

### Data
- Chronofast study by Deutsches Institut für Ernährungsforschung Potsdam-Rehbrücke (DIfE)
- Parofastin study by  Deutsches Institut für Ernährungsforschung Potsdam-Rehbrücke (DIfE) & Department of Periodontology, Oral Medicine and Oral Surgery of Charité Centrum in Berlin
- Simulated Glucose data by simglucose (https://github.com/jxx123/simglucose)

### Training on labeled glucose and acceleration data
Executed in Jupyter Notebooks

- Via Hutchison Heuristic
J. Hutchinson, Joseph Alba, and Eric Eisenstein. Heuristics and Biases in Data-Based Decision Making: Effects of Experience, Training, and Graphical Data Displays. Journal of Marketing Research- J MARKET RES-CHICAGO 47 (Aug. 2010), 627–642. doi:10.1509/jmkr.47.4.627 (see pages 22, 69).
- Via Machine Learning
- Via Time Series Classification

### Dashboard Implementation

Dash application that runs local on http://127.0.0.1:8050/

For starting the dashboard, open shell in the directory.

```sh
python main.py
```
