# GeoFM Crop Type Classification

## Overview

Geospatial Foundation Models (GeoFMs) are powerful deep learning models trained on massive and diverse Earth observation data such as satellite imagery, elevation maps, and remote sensing bands. These models can be fine-tuned or used directly for many geospatial tasks like:

- Land use classification
- Crop yield prediction
- Climate pattern analysis
- Disaster detection
- Urban planning

In this challenge, the goal is to classify **crop types** using **pixel-level time series** data from Sentinel-2 multispectral satellite imagery.

### Why it matters

Accurate crop type classification using satellite data helps reduce the need for manual ground surveys, which are time-consuming and expensive. Faster and automated labelling enables quicker yield estimation, better policy-making, and more efficient agricultural assistance programs.

---

## Task

You are provided with:

- Time-series spectral band data (e.g., red, blue, nir, swir)
- Ground-truth crop type labels for training
- Unlabeled test data

Your job is to use one (or both) of the following GeoFMs:

1. **PRESTO** – from NASA Harvest, trained on Sentinel-1 & Sentinel-2
2. **Amini Earth FM** – trained on Sentinel-2 data sampled across Africa

Then, generate **embeddings** for each pixel time series and train a classifier to predict one of the crop types:

- `cocoa`
- `rubber`
- `oil`

---

## Model & Approach

This solution uses the **Amini Earth FM** model: `AminiTech/fm-v2-28M`.

### Steps:

1. **Preprocess raw satellite data**
   - Compute NDVI and NDWI indices
   - Interpolate all bands to a fixed length (48 steps)
   - Compute summary stats (mean, std) per pixel

2. **Embed pixel time-series**
   - Load the Amini FM model
   - Use it to generate CLS and mean embeddings for each sample

3. **Combine features**
   - Concatenate embeddings + NDVI/NDWI stats
   - Apply standard scaling and PCA for dimensionality reduction

4. **Train classifier**
   - Use an XGBoost classifier wrapped in `CalibratedClassifierCV`
   - Train using Stratified K-Fold cross-validation (5 folds)

5. **Generate predictions**
   - Output probabilities for each class on the test set
   - Save results in required submission format

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas torch scikit-learn xgboost transformers tqdm scipy
```
---
```txt
A Little Note from Me :)

So here's the deal — I scored 80 out of 168, which means I'm in the top 47%.
Not gonna lie, it's not as good as my last one.

But I have two solid (and totally valid) reasons for that:
1. I had my improvement exam — so yeah, study pressure was real.
2. I had a medical issue — couldn’t perform at full speed like I usually do.

Still, I gave it my best under the circumstances. You can totally expect me
to bounce back stronger in upcoming events 

Thanks for checking out my work!

P.S. You can get the dataset from 👉 https://zindi.africa
```
---
### Here is a proof
<img width="1365" height="727" alt="image" src="https://github.com/user-attachments/assets/1eebeedc-c502-479f-886c-b73565d77f05" />
---
### Here is  a dataset intro for you, Thanks me later

## 📂 Dataset Files

| File Name               | Size     | Description                                                                 |
|------------------------|----------|-----------------------------------------------------------------------------|
| `test.csv`             | 154.6 MB | Contains the test data without target labels. Your model makes predictions on this. |
| `SampleSubmission.csv` | 164.5 KB | Example submission format. Make sure your `unique_id` column matches!       |
| `StarterNotebook.ipynb`| 58.3 KB  | A notebook showing how to use GeoFM. Helpful for getting started.           |
| `dummy_satellite_data.csv` | 3.5 MB  | A small, fake dataset for testing and understanding the pipeline. Not used for actual training. |
---
### Sayonaraa
