# <div align="center"> Smart Product Pricing Challenge
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Llama 3](https://img.shields.io/badge/LLM-Meta_Llama_3-purple?style=for-the-badge&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()
</div>

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 75k products with complete product details and prices
- **Test Set:** 75k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.


### File Descriptions:

*Source files*

1. **src/utils.py:** Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
2. **sample_code.py:** Sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
3. **dataset/sample_test.csv:** Sample test input file.
4. **dataset/sample_test_out.csv:** Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120  
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

> A multi-modal hybrid architecture that combines statistical regression (XGBRegressor) with semantic reasoning (LLM) to predict retail product prices with high accuracy.

---

## 📌 Overview

This project implements a **Multi-Modal Hybrid Architecture** to predict retail product prices using **catalog text, product images, and numerical metadata**.  
The system combines the statistical strength of **XGBoost** with the semantic reasoning ability of a **Large Language Model (Meta-Llama-3-8B-Instruct)**.

Instead of replacing classical ML, the LLM is used as a **logic-constrained post-processor** that corrects large magnitude errors—especially for bulk and multi-pack products.

 **Result:** Reduced SMAPE from **~82% (XGB only)** to **~67% (Hybrid)**  
 **Competitive Range:** 50%–70% SMAPE (as per challenge leaderboard)

---

## 🚀 Key Idea

> **Use ML for precision, LLM for reasoning.**  
> The regressor predicts *how much* based on data patterns,  
> the LLM decides *whether that magnitude makes sense*.

---

## 🛠️ Technical Stack

| Category | Technologies |
|--------|--------------|
| Core Language | Python 3.10+ |
| Machine Learning | XGBoost (Regressor), Scikit-Learn |
| Deep Learning / LLM | PyTorch, Hugging Face Transformers |
| Models | **Base:** XGBRegressor<br>**Refiner:** Meta-Llama-3-8B-Instruct |
| Data Processing | Pandas, NumPy, Regex-based text parsing |
| Vision | Precomputed image embeddings (zero-padded for missing images) |
| Hardware | Google Colab T4 GPU (16GB VRAM) |

---

## 🔬 Methodology

### **Phase 1: Feature Engineering & Fusion**

**Preprocessing**
- Parsed semi-structured catalog content
- Extracted:
  - Pack size
  - Units (g, ml, oz, etc.)
  - Total quantity
- Applied log transformations to stabilize scaling

**Embeddings**
- **Text:** Dense semantic embeddings from product descriptions
- **Vision:** Image embeddings (missing images handled via zero-padding)

**Feature Fusion**
- Numerical + Text + Vision features concatenated into a single matrix

**Weighted Scaling**
To control feature dominance:
- Numerical → **35%**
- Textual → **45%**
- Visual → **20%**

This prevents noisy images from overwhelming reliable signals.

---

### **Phase 2: Statistical Prediction (Base Model)**

- **Model:** `XGBRegressor`
- **Target Transformation:** `log1p(price)`
- **Objective:** Learn price elasticity and attribute correlations
- **Output:** Base statistical price prediction

---

### **Phase 3: Semantic Refinement (LLM Logic Layer)**

#### 🔴 Problem
Tree models struggle to distinguish:
- **Single unit vs Bulk cases**
- Similar embeddings → massive price underestimation
- Example: predicting `$5` for a `$50` bulk case → catastrophic SMAPE

#### 🟢 Solution
Use **Meta-Llama-3-8B-Instruct** as a **constrained classifier**, NOT a generator.

**LLM Role**
- Classifies quantity tier only
- Never generates prices
- Deterministic multiplier applied post-classification

---

## 🏷️ LLM Calibration Logic

| LLM Tag | Quantity Definition | Multiplier |
|-------|---------------------|------------|
| `TAG_SINGLE` | Single unit | **1.45×** |
| `TAG_PACK_SMALL` | 2–10 items | **3.9×** |
| `TAG_BULK_HUGE` | Industrial / Pallet (50+) | **6.7×** |

This **logic-over-generation** approach prevents hallucinations and stabilizes error metrics.

---

## ✨ Key Innovations

1. **Manual Feature Power Balancing**  
   Controlled signal dominance using variance-based weighting.

2. **LLM as a Logic Layer (Not a Predictor)**  
   Restricting the LLM to classification avoids unstable price generation.

3. **Multi-Tier Quantity Reasoning**  
   Going beyond binary bulk detection captures real-world pricing nuances.

4. **Serial Ensemble Design**  
   Combines speed of ML with reasoning depth of LLMs.

---

## 📊 Final Results

| Model | SMAPE |
|-----|-------|
| XGBoost Only | ~82% |
| Hybrid (XGB + Llama-3) | **~67%** |

✔ Achieved **competition-viable performance**  
✔ Significant reduction in catastrophic pricing errors  

---

## 📦 Saved Artifacts

- `xgb_price_model.pkl` – Base regressor
- `feature_scaler.pkl` – Weighted scaler
- `weight_factors.pkl` – Feature importance logic
- `final_aggressive_submission.csv` – Final predictions

---

## 🔮 Future Improvements

- Learn multipliers automatically via calibration layer
- Replace rule-based scaling with differentiable gating
- Add uncertainty-aware confidence thresholds
- Train a lightweight classifier to replace LLM inference

---

## 🏁 Conclusion

This project demonstrates how **classical ML and LLMs can work together**:
- ML handles regression with precision
- LLMs inject semantic reasoning
- The result is a **stable, interpretable, and competitive pricing system**

---

> *“LLMs shouldn’t replace models — they should reason about them.”*
