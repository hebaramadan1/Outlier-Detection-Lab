
# Outlier Detection & Treatment Lab

> An interactive Data Science application for detecting, visualizing, and exploring the treatment of outliers in real-world datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-red)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-orange)

<br>

# 📸 Application Preview

<!-- Replace these paths with your actual image paths -->

<p align="center">
  <img src="images/screenshot 1.png" width="100%" alt="Main Dashboard">
</p>

<p align="center">
  <img src="images/screenshot 2.png" width="100%" alt="Outlier Detection Visualization">
</p>

<p align="center">
  <img src="images/screenshot 3.png" width="100%" alt="Research Guide">
</p>

---

# 📌 Overview

**Outlier Detection & Treatment Lab** is an interactive Data Science application designed to help users explore, detect, visualize, and understand potential outliers in real-world datasets.

Outliers are observations that differ significantly from the majority of the data. While some outliers may result from data entry or measurement errors, others may represent meaningful and important observations.

For this reason, the application does not treat outlier detection as an automatic data deletion process. Instead, it provides an interactive environment where users can:

- Upload and analyze CSV datasets.
- Use the default Heart Disease dataset.
- Select numerical features for analysis.
- Detect potential outliers using the IQR method.
- Visualize data distributions.
- Explore statistical boundaries.
- Apply different treatment strategies.
- Learn about the statistical concepts behind outlier detection.

The project combines **Statistical Analysis, Data Cleaning, Visualization, and Interactive Web Development** in one application.

---

# 🎯 Problem Statement

Real-world datasets often contain extreme observations that can significantly influence statistical analysis.

Traditional statistical measures such as the **mean** and **standard deviation** can be affected by unusually large or small observations.

However, removing every detected outlier is not always the correct solution.

An outlier may represent:

- ❌ A data entry error.
- 📏 A measurement error.
- ⚠️ An unusual observation.
- 🧬 A rare but valid case.
- ❤️ An important clinical condition.

Therefore, the main challenge is not only:

> **How can we detect outliers?**

But also:

> **How should we interpret and treat them?**

This project provides an interactive environment for exploring this problem.

---

# 🔬 Research Question

> **How can different outlier detection and treatment strategies influence the statistical interpretation of real-world data?**

Using the **Heart Disease dataset** as a case study, the project explores how potential outliers can be identified and how different treatment approaches can be applied.

---

# ✨ Features

## 📂 Dataset Upload

Users can upload their own datasets in **CSV format**.

The application automatically processes the uploaded dataset and allows numerical features to be selected for analysis.

The project also includes a **default Heart Disease dataset** for demonstration.

---

## 🎛️ Interactive Control Panel

The Streamlit sidebar allows users to control the analysis.

Users can:

- 📁 Upload a custom CSV dataset.
- ❤️ Use the default Heart Disease dataset.
- 🔢 Select a numerical feature.
- 🔍 Select an outlier detection method.
- 🛠️ Select an outlier treatment strategy.

---

# 🔍 Outlier Detection

## 📊 IQR Method

The application uses the **Interquartile Range (IQR)** methodology to identify potential outliers.

The Interquartile Range is calculated using:

```text
IQR = Q3 - Q1