---
theme: unicorn
highlighter: shiki

info: |
  ## Is Your ML Model Trustworthy?
  by María Grandury

  [My personal website](https://mariagrandury.github.io)

layout: center
logoHeader: 'https://mlopsworld.com/wp-content/uploads/2020/04/MLOps-Logo-2-150x150.png'
website: 'mariagrandury.github.io'
handle: 'mariagrandury'
---

<a href="https://github.com/mariagrandury" target="_blank" alt="GitHub"
  class="abs-tr m-6 text-3xl icon-btn opacity-50 !border-none !hover:text-white">
  <carbon-logo-github />
</a>

<div class="place-items-center">

# Is Your ML Model Trustworthy?

## María Grandury

</div>

---
handle: 'mariagrandury'
layout: intro
introImage: 'https://avatars.githubusercontent.com/u/57645283?v=4'
---

<a href="https://github.com/mariagrandury" target="_blank" alt="GitHub"
  class="abs-tr m-6 text-3xl icon-btn opacity-50 !border-none !hover:text-white"> <carbon-logo-github />
</a>

# About me

- 💡  Machine Learning Research Engineer

- 🎯  **#NLP**, AI Robustness & Explainability (**#XAI**)

- 🎓  Mathematician & Physicist

- 👩🏻‍💻  Trusted AI [**@neurocat.ai**](https://www.neurocat.ai/)

- 🚀  Founder [**@NLP_en_ES**](https://twitter.com/nlp_en_es) 🤗

- ⚡  Core Team [**@WAIRobotics**](https://twitter.com/wairobotics)

---
handle: 'mariagrandury'
---

# Functionality

<div grid="~ cols-2 gap-4">

<div>

<img style="height: 350px; margin-left: 50px" class="rounded" src="ml_flow.png">

</div>

<div>

## Performance Metrics
- Classification: Accuracy, F1-score, AUC
- Regression: MSE, MAE, R²
- Ranking: Mean Reciprocal Rank
- NLP: BLEU, ROUGE, Perplexity
- GANs: Inception score, Frechet Inception distance

<br>
<br>

<v-click>
<div>

## THE Question

🔴 "Is my performance metric high enough?"

✅ "Is my model trustworthy enough?"

</div>
</v-click>

</div>
</div>

---
handle: 'mariagrandury'
---

# Explainability

<div grid="~ cols-1" class="place-items-center">
<img style="height: 350px; margin-left: 50px" class="rounded" src="https://www.researchgate.net/profile/Michael-Wade-5/publication/302632920/figure/fig2/AS:751645805789184@1556217733527/Then-a-Miracle-Occurs-Copyrighted-artwork-by-Sydney-Harris-Inc-All-materials-used-with.png">
</div>

---
handle: 'mariagrandury'
---

# Explaining ML Models with SHAP

<div grid="~ cols-2 gap-12">

<div>

<br>

📄 [Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (2018)](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

<br>

⭐ [github.com/slundberg/shap](https://github.com/slundberg/shap) (12.9k)

<br>

- SHapley Additive exPlanations
- game theoretic approach
- explain the output of any ML model

</div>

<div>

<img style="height: 300px; margin-left: -50px" class="rounded" src="https://shap.readthedocs.io/en/latest/_images/shap_header.png">

</div>
</div>

---
handle: 'mariagrandury'
---

# Explainability with SHAP - Tabular Data
 
<div grid="~ cols-2 gap-8">

<div>
<!-- The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue. -->

The features contribute to push the model output from the base value to the model output:

🔴 Push the prediction higher 

🔵 Push the prediction lower

<br>

Example:

- Boston Housing data set
- Regression Model
- XGBoost

</div>

<div>

<img style="height: 300px; margin-left: -30px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_waterfall.png">

</div>
</div>

---
handle: 'mariagrandury'
---

# Explainability with SHAP - NLP

<div grid="~ cols-2 gap-4">

<div>

- Force plot
- [Lundberg, Lee et al., "Explainable ML predictions for the prevention of hypoxaemia during surgery" (2018)](https://www.nature.com/articles/s41551-018-0304-0.epdf?author_access_token=vSPt7ryUfdSCv4qcyeEuCdRgN0jAjWel9jnR3ZoTv0PdqacSN9qNY_fC0jWkIQUd0L2zaj3bbIQEdrTqCczGWv2brU5rTJPxyss1N4yTIHpnSv5_nBVJoUbvejyvvjrGTb2odwWKT2Bfvl0ExQKhZw%3D%3D)

</div>

<div>

- IMDb data set
- Sentiment Analysis
- 🤗 Transformers
- Explanation for the POSITIVE output class

</div>

</div>

<br>
<br>

<img style="width: 1000px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/sentiment_analysis_plot.png">

---
handle: 'mariagrandury'
---

# Explainability with SHAP - Computer Vision

<div grid="~ cols-2 gap-4">

<div>

- MNIST data set
- Classification Model

</div>

<div>

<img style="width: 30000px; height: 350px; margin-left: -200px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/mnist_image_plot.png">

</div>
</div>

---
layout: center
logoHeader: 'https://mlopsworld.com/wp-content/uploads/2020/04/MLOps-Logo-2-150x150.png'
website: 'mariagrandury.github.io'
handle: 'mariagrandury'
---

<div class="place-items-center">

# Thank you!

<br>

Slides can be found at [github.com/mariagrandury](https://github.com/mariagrandury)

</div>

---
handle: 'mariagrandury'
---

# More Resources

## Explainability:

- C. Molnar, "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019, [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/)
- A. Saucedo, "Guide towards algorithm explainability in machine learning", [talk at PyData London 2019](https://www.youtube.com/watch?v=vq8mDiDODhc)
- S. Lundberg, "Explainable Machine Learning with Shapley Values", [talk at #H2OWorld 2019](https://www.youtube.com/watch?v=ngOBhhINWb8)
- A. Chouldechova and A. Roth, “A snapshot of the frontiers of fairness in machine learning,” Commun. ACM, vol. 63, no. 5, pp. 82–89, Apr. 2020, [doi: 10.1145/3376898](https://dl.acm.org/doi/10.1145/3376898)
- V. Dignum, "The Mith of Complete AI Fairness", 2021, [arXiv:2104.12544v1 [cs.CY]](https://arxiv.org/pdf/2104.12544.pdf)

## Adversarial Attacks:

- "Attacking Machine Learning with Adversarial Examples" - [OpenAI Blog Post](https://openai.com/blog/adversarial-example-research/)
- FGSM with TensorFlow2 - [TF Tutorial](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
- J. Morris, "TextAttack: A Framework for Data Augmentation and Adversarial Training in NLP", [talk at dair.ai](https://www.youtube.com/watch?v=VpLAjOQHaLU)

---
handle: 'mariagrandury'
---

# Code: SHAP & Tabular Data

<div grid="~ cols-2 gap-4">

<div>

```py
!pip install shap

import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
```

</div>
<div>
<img style="height: 300px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_waterfall.png">
</div>

</div>

---

# Code: SHAP & NLP

```py
import transformers
import shap

# load a transformers pipeline model
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on two sample inputs
explainer = shap.Explainer(model) 
shap_values = explainer(["What a great movie! ...if you have no taste."])

# visualize the first prediction's explanation for the POSITIVE output class
shap.plots.text(shap_values[0, :, "POSITIVE"])
```

<br>

<img style="width: 1000px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/sentiment_analysis_plot.png">

---
handle: 'mariagrandury'
---

# Code: SHAP & CV

<div grid="~ cols-2 gap-4">

<div>

```py
import shap
import numpy as np

# select a set of background examples to take an
# expectation over
background = x_train[np.random.choice(
  x_train.shape[0], 100, replace=False
)]

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(x_test[1:5])

# plot the feature attributions
shap.image_plot(shap_values, -x_test[1:5])
```
</div>
<div>
<img style="height: 300px; margin-left: -30px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/mnist_image_plot.png">
</div>
</div>

---
handle: 'mariagrandury'
---

# How to improve the performance of my ML Model?

<div grid="~ cols-2 gap-4">

<div>

If bias == high => 
- Bigger NN (size of hidden units, number of layers)
- Train longer
- Different optimization algorithms

<br>

If variance == high (i.e. overfitting) =>
- More data (data augmentation)
- Regularization (L2, dropout)

<!-- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/2-%20Improving%20Deep%20Neural%20Networks#bias--variance -->

</div>

<div>

<br>
<br>

<a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff">
<img style="height: 200px" class="rounded" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/1024px-Bias_and_variance_contributing_to_total_error.svg.png"></a>

</div>

</div>
