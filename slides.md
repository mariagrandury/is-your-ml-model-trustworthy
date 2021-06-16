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

## Performance Metrics
- Classification:
  - Accuracy, Precision, Recall, F1-score
  - ROC, AUC
- Regression:
  - MSE, MAE
  - Pearson CC, R²
- Ranking: Mean Reciprocal Rank
- NLP:
  - BLEU, ROUGE
  - Perplexity
- GANs:
  - Inception score, Frechet Inception distance

</div>

<div>

<img style="height: 400px; margin-left: 50px" class="rounded" src="ml_flow.png">

</div>

</div>

<!-- 

https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce 

https://towardsdatascience.com/the-most-common-evaluation-metrics-in-nlp-ced6a763ac8b

-->

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

📄 [Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions"](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) (2017)

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
- [Lundberg, Lee et al., "Explainable ML predictions for the prevention of hypoxaemia during surgery"](https://www.nature.com/articles/s41551-018-0304-0.epdf?author_access_token=vSPt7ryUfdSCv4qcyeEuCdRgN0jAjWel9jnR3ZoTv0PdqacSN9qNY_fC0jWkIQUd0L2zaj3bbIQEdrTqCczGWv2brU5rTJPxyss1N4yTIHpnSv5_nBVJoUbvejyvvjrGTb2odwWKT2Bfvl0ExQKhZw%3D%3D) (2018)

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

<br>

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
