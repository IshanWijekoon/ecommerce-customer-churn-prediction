# Viva Preparation Guide

Use this document before and during the project viva. Speak in plain English.
Do not memorise code — understand the *why*.

---

## 1. Project summary (30-second version)

> We predict e-commerce customer churn. Only about 17% of customers leave, so
> accuracy alone is misleading. We clean the data, explore risk patterns,
> train simple baselines, use a Genetic Algorithm and PSO to select features,
> train TabNet on the winning subset, then explain the drivers with permutation
> importance and turn them into retention actions.

---

## 2. Data flow diagram

```text
Kaggle Excel
   data/raw/E_Commerce_Dataset.xlsx
              |
              v
   [1] Data Understanding
              |  step1_customers.csv
              v
   [2] Data Exploration (EDA + K-means)
              |  insights / hypotheses
              v
   [3] Preprocessing (fill, encode, scale, split)
              |  X_train, X_test, y_train, y_test
              v
   [4] Baselines (LR + Random Forest)
              |  fitness model choice
              +------------------+
              |                  |
              v                  v
        [5] Genetic Algorithm   [6] PSO
              |                  |
              +--------+---------+
                       |
                       v
              winning_selected_features.csv
                       |
                       v
              [7] TabNet (winning subset)
                       |
                       v
              [8] Permutation importance
                  + risk bands
                  + retention plays
```

---

## 3. Execution flow (start to finish)

1. Place Excel in `data/raw/`
2. Run notebook 1 → save clean customer table
3. Run notebook 2 → find churn patterns
4. Run notebook 3 → create model-ready train/test matrices
5. Run notebook 4 → baselines + choose RF as fitness model
6. Run notebook 5 → GA feature subset
7. Run notebook 6 → PSO subset + pick winner (GA on this run)
8. Run notebook 7 → TabNet predictions + importances
9. Run notebook 8 → explain drivers + business actions

---

## 4. How to explain the project in simple English

Use this script if you freeze:

1. **Problem:** Which customers will leave, and why?
2. **Data:** 5,630 customers, 20 columns, ~17% churn.
3. **Challenge:** Imbalanced data → do not trust accuracy alone.
4. **EDA:** Complaints, short tenure, low satisfaction raise churn risk.
5. **Preprocessing:** Fix labels, fill blanks with train medians, one-hot encode, RobustScaler, 80/20 stratified split.
6. **Baselines:** Logistic Regression is linear; Random Forest is stronger.
7. **NIA:** GA and PSO search for smaller useful feature sets using CV F1.
8. **TabNet:** Neural net for tables; gives strong probabilities.
9. **Explain:** Shuffle a feature; if score drops, it mattered.
10. **Action:** High-risk customers get priority CX / win-back.

---

## 5. Top 20 viva questions (with short answers)

### Basics

**1. What is churn?**  
A customer stops buying and does not return. In our data, `Churn=1` means left.

**2. Why is this problem important?**  
Retaining customers usually costs less than acquiring new ones.

**3. What type of ML problem is this?**  
Binary classification.

**4. Why not use Accuracy as the main metric?**  
A model that always predicts “stay” gets ~83% accuracy but catches zero churners.

**5. What are Precision and Recall?**  
Precision: of flagged churners, how many truly left.  
Recall: of true churners, how many we caught.

### Data & preprocessing

**6. How did you handle missing values?**  
Median for numbers, mode for categories, learned from **train only**.

**7. What is data leakage?**  
Using test information while preparing train data, which makes test scores unrealistically good.

**8. Why drop CustomerID?**  
It identifies a person; it is not a behavioural predictor.

**9. What is one-hot encoding?**  
Turning each category into its own 0/1 column.

**10. Why RobustScaler?**  
It uses median and IQR, so outliers affect scaling less than StandardScaler.

### Models & NIA

**11. Why Logistic Regression and Random Forest?**  
Simple baselines: one linear, one non-linear ensemble.

**12. What is a Genetic Algorithm here?**  
Each chromosome is a 0/1 feature mask. Selection, crossover, mutation search for high-F1 subsets.

**13. What is the fitness function?**  
CV F1 of a small Random Forest minus a small penalty for using many features.

**14. What is PSO here?**  
Particles move in continuous space toward personal and global bests; sigmoid turns positions into feature masks.

**15. Who won GA vs PSO, and why does that matter?**  
GA won on this run (higher CV F1, fewer features). TabNet trains on that winning list.

**16. What is TabNet?**  
A neural network for tabular data that attends to important features in steps.

**17. What does P(churn)=0.9 mean?**  
The model thinks there is a high chance this customer will leave.

### Explainability & business

**18. What is permutation importance?**  
Shuffle one feature; the drop in score measures importance.

**19. What should the business do with high-risk scores?**  
Priority outreach: fix complaints first, then onboarding / win-back offers.

**20. What is overfitting?**  
Memorising train data and failing on new customers. We limit tree depth and always check a held-out test set.

---

## 6. Harder follow-ups (be ready)

- **False Negative vs False Positive:** FN (miss a churner) is usually costlier.
- **Why CV inside GA/PSO, not the test set?** Test must stay clean for final evaluation.
- **Why GA subset can lose slightly to all-features TabNet?** Different model family; subset is still strong and easier to explain.
- **Could DaySinceLastOrder leak the label?** Only if churn were defined from recency. Here churn is a separate flag, so we keep it but watch its importance.

---

## 7. Common mistakes students make

1. Saying “our accuracy is high” without mentioning imbalance  
2. Cannot explain why train statistics must not use the test set  
3. Describing GA/PSO as “the AI library did it” instead of selection/crossover/velocity  
4. Confusing clustering (groups) with classification (individual prediction)  
5. Reading a chart without giving a business conclusion  
6. Memorising parameter names without meanings (`n_steps`, `class_weight`, etc.)  
7. Claiming SHAP when the project uses permutation importance  

---

## 8. Five-member presentation split

Balanced workload for a ~20–25 minute viva + demos.

| Member | Owns | Notebooks | Must be able to explain |
|--------|------|-----------|-------------------------|
| **1 — Problem & Data** | Business problem, dataset, quality, imbalance, metrics choice | 1 | Why recall > accuracy; missing-value decision |
| **2 — EDA & Insights** | Charts, associations, tenure cohorts, K-means segments | 2 | Complain / tenure / satisfaction story; clustering vs classification |
| **3 — Preprocessing & Baselines** | Cleaning pipeline, leakage checklist, LR vs RF | 3, 4 | Train-only fills; confusion matrix; F1/PR-AUC |
| **4 — Nature-Inspired Algorithms** | GA, PSO, fitness, winner rule | 5, 6 | Chromosome/mask; velocity update; why same fitness |
| **5 — TabNet, Prediction & Business** | TabNet, permutation importance, risk bands, retention plays | 7, 8 | P(churn); why shuffle explains drivers; action plan |

### Suggested speaking order

1. Member 1 (3–4 min)  
2. Member 2 (4–5 min)  
3. Member 3 (4–5 min)  
4. Member 4 (5–6 min) — course core  
5. Member 5 (4–5 min) — prediction + business close  

Everyone should still know the 30-second summary and the top-20 answers at a basic level.

---

## 9. Demo checklist (day before viva)

- [ ] Excel is in `data/raw/E_Commerce_Dataset.xlsx`
- [ ] Environment activates (`ecommerce-churn`)
- [ ] Notebooks 1–8 open without missing-file errors
- [ ] Know where these charts are: churn balance, GA fitness, GA vs PSO, TabNet importance, risk bands, leaderboard
- [ ] Each member practised their section out loud once without notes

---

## 10. One closing sentence

> Our project does not only predict churn — it selects features with nature-inspired
> algorithms, explains the drivers simply, and turns scores into retention actions.
