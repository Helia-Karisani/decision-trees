# Decision Tree Classification (Entropy-Based)

This project implements a **Decision Tree Classifier** to perform **multi-class classification**.  
The model learns a hierarchy of decision rules that split the data based on feature values in order to predict a target class.

Unlike linear models that learn coefficients, **the tree structure itself is the learned model**.

---

## Problem Type

- **Learning type**: Supervised learning
- **Task**: Multi-class classification
- **Model**: Decision Tree Classifier
- **Splitting criterion**: Entropy (Information Gain)

---

## Dataset Overview

- Rows represent individual samples
- Columns represent input features
- One column represents the **target class**

The goal is to predict the class label using the input features by learning a sequence of decision rules.

---

## What Is a Decision Tree?

A decision tree is a **rule-based model** that predicts a class by asking a sequence of yes/no questions.

A decision tree is a rule-based learning model, it learns a sequence of IF–THEN rules.
During training, the algorithm learns which questions to ask, and in what order. 
The algorithm does something brute-force, comparing all possible values of all features. 
A decision tree is built by repeatedly separating the training rows using the single question that most reduces class uncertainty, until no useful separation is possible.

Example logic:
```
IF feature_4 <= 14.0:
    IF feature_2 <= 0.5:
        predict Class A
    ELSE:
        predict Class B
ELSE:
    predict Class C
```

Each internal node is a **decision rule**, and each leaf node is a **final prediction**.

---

## How the Tree Is Built (Training Phase)

The tree is constructed **top-down**, using a greedy algorithm.

### Step 1: Start with all training samples

At the root node:
- All training rows are grouped together
- Class labels are mixed
- Uncertainty is high

---

### Step 2: Measure uncertainty using entropy
The decision tree chooses its splits by measuring and reducing uncertainty using entropy:

-High entropy → classes are mixed → high uncertainty
-Low entropy → one class dominates → low uncertainty
-Entropy = 0 → all samples belong to one class (perfectly pure)

Entropy measures how mixed the classes are:

H = - Σ p_i log2(p_i)

where p_i is the proportion of class i in the node.

- High entropy → mixed classes
- Entropy = 0 → all samples belong to one class

---

### Step 3: Try all possible splits

For each feature:
- Try multiple threshold values
- For each candidate split:
  - Split rows into left/right groups
  - Compute entropy for each group
  - Compute weighted entropy

---

### Step 4: Choose the best split (Information Gain)

Information Gain measures how much uncertainty is reduced:

IG = H_parent − (n_left / n) * H_left − (n_right / n) * H_right

The split with **maximum Information Gain** is selected.

This produces a rule like:
```
feature_k <= threshold
```

---

### Step 5: Split the data

- Samples satisfying the condition go to the **left child**
- The remaining samples go to the **right child**
- No data is lost — samples are partitioned

---

### Step 6: Repeat recursively

Steps 2–5 are repeated independently on each child node.

---

### Step 7: Stopping conditions

A node becomes a **leaf** if:
- Entropy = 0 (pure node)
- Maximum depth is reached
- No split improves information gain
- Too few samples remain

---

## What Each Node Represents

Each node in the trained tree shows:

- **Decision rule**  
  Example: `x[4] <= 14.027`

- **Entropy**  
  Remaining uncertainty at that node

- **Samples**  
  Number of training rows that reached this node

- **Value**  
  Class counts at this node  
  Example: `[15, 11, 13, 38, 63]`

Prediction at a node is the **class with the highest count**.

---

## Leaf Nodes (Final Decisions)

A leaf node represents a final prediction.

Example:
```
entropy = 0.0
samples = 28
value = [0, 0, 0, 28, 0]
```

Interpretation:
- All 28 samples belong to the same class
- The model stops splitting
- Any new sample reaching this leaf is predicted as that class

---

## What the Tree Splits (Important Clarification)

The tree **splits data points (rows)** — not features, not classes.

At each node:
- Training rows are divided into two groups
- Each row follows exactly one branch
- The process continues until a leaf is reached

---

## Prediction Phase (After Training)

To classify a new sample:

1. Start at the root
2. Evaluate the decision rule
3. Move left or right
4. Repeat until a leaf is reached
5. Output the leaf’s class

No training samples are involved during prediction — only the learned rules.

---

## Model Configuration Used

- Criterion: `entropy`
- Maximum depth: controlled to prevent overfitting
- Algorithm: greedy, top-down recursive splitting

---

## Why Decision Trees Are Learning Models

Decision trees learn:
- Which features matter
- Which thresholds to use
- In what order decisions should be made

The learned **tree structure is the model**.

---

## Key Characteristics

- Interpretable
- Non-linear
- Handles mixed feature types
- No feature scaling required
- Prone to overfitting without depth control

---

## Summary

This project demonstrates:
- How a decision tree is built from data
- How entropy and information gain guide learning
- How classification is performed using learned rules
- Why the tree visualization is the trained model itself

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib (for visualization)
