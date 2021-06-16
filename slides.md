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

# Explainability

<br>

XAI plays a crucial role in sensitive domains.

<br>

<div grid="~ cols-2 gap-10" class="place-items-center">
<img style="height: 250px; margin-left: 50px" class="rounded" src="https://images.unsplash.com/photo-1505751172876-fa1923c5c528?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=750&q=80">


<img style="height: 250px" class="rounded" src="https://images.unsplash.com/photo-1471864190281-a93a3070b6de?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=750&q=80">
</div>

---
handle: 'mariagrandury'
---

# Explaining ML Models with SHAP

<div grid="~ cols-2 gap-12">

<div>

<br>

📄 Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions", 2017, [NIPS](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

<br>

⭐ [github.com/slundberg/shap](https://github.com/slundberg/shap) (12.9k)

<br>

- SHapley Additive exPlanations
- game theoretic approach
- explain the a prediction by computing the contribution of each feature to the prediction

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

The features contribute to push the model output from the base value to the model output:

🔴 Push the prediction higher 

🔵 Push the prediction lower

<br>

Example:

- [Boston Housing data set](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
  - LSTAT: % lower status of the population
  - RM: average number of rooms per residence
- Regression Model (XGBoost)

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
- Lundberg, Lee et al., "Explainable ML predictions for the prevention of hypoxaemia during surgery", 2018, [doi: 10.1038/s41551-018-0304-0](https://doi.org/10.1038/s41551-018-0304-0)

</div>

<div>

- [IMDb movie review data set](https://ai.stanford.edu/~amaas/data/sentiment/)
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

- [MNIST data set](http://yann.lecun.com/exdb/mnist/)
- Classification Model

</div>

<div>

<img style="width: 30000px; height: 350px; margin-left: -200px" class="rounded" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/mnist_image_plot.png">

</div>
</div>

---
handle: 'mariagrandury'
---

# Robustness

<br>

<div grid="~ cols-3 gap-4" class="place-items-center">
<img style="height: 250px" class="rounded" src="https://images.unsplash.com/photo-1612101983844-b37abf2b17f7?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=80">

<img style="height: 250px; margin-left: -80px" class="rounded" src="https://images.unsplash.com/photo-1612725018378-00328e597d9a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=334&q=80">


<img style="height: 250px; margin-left: -80px" class="rounded" src="https://images.unsplash.com/photo-1594482628048-53865e5a59c4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=667&q=80">
</div>

---
handle: 'mariagrandury'
---

# Adversarial Attacks

📄 Szegedy, “Intriguing properties of neural networks”, 2013, [arXiv:1312.6199 [cs.CV]](https://arxiv.org/abs/1312.6199)

<br>

<div grid="~ cols-2 gap-20">

<div>

📄 Goodfellow, Shlens and Szegedy, "Explaining and Harnessing Adversarial Examples", 2014, [arXiv:1412.6572 [stat.ML]](https://arxiv.org/abs/1412.6572)

<img class="rounded" src="https://www.tensorflow.org/tutorials/generative/images/adversarial_example.png">

</div>

<div>

<figure>
<img style="height: 200px; margin-left: 40px;" class="rounded" src="https://nicholas.carlini.com/writing/2019/advex_plot.png">
<figcaption style="font-size: small">A Complete List of All (arXiv) Adversarial Example Papers
by Nicholas Carlini.</figcaption>
</figure>

</div>
</div>

---
handle: 'mariagrandury'
---

# Adversarial Attacks with CleverHans

<div grid="~ cols-2 gap-4">

<div>

📄 Papernot, Faghri, Carlini, Goodfellow et al., "Technical Report on the CleverHans v2.1.0 Adversarial Examples Library", 2018, [arXiv:1610.00768 [cs.LG]](https://arxiv.org/abs/1610.00768)

<br>

⭐ [github.com/cleverhans-lab/cleverhans](https://github.com/cleverhans-lab/cleverhans) (5.1k)
- Attacks: [FGSM Attack](https://arxiv.org/abs/1412.6572), [Carlini Wagner Attack](https://arxiv.org/abs/1608.04644)
- Defenses: [Resampling](https://archive.nyu.edu/handle/2451/60767)

<br>
<br>

✏️ [CleverHans Blog](https://www.cleverhans.io/)

</div>

<div>

<img style="height: 300px; margin-left: 80px;" class="rounded" src="https://secml.github.io/images/class3/class3_img1.png">

</div>

</div>

---
handle: 'mariagrandury'
---

# Robustness - NLP & TextAttack

<div grid="~ cols-2 gap-4">

<div>

📄 Morris et al., "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP", 2020, [	arXiv:2005.05909 [cs.CL]](https://arxiv.org/abs/2005.05909)

<br>

⭐ [github.com/QData/TextAttack](https://github.com/QData/TextAttack) (1.5k)


[All attack recipes](https://textattack.readthedocs.io/en/latest/3recipes/attack_recipes.html)
</div>

<div>
<img style="height: 150px; margin-left: 0px" class="rounded" src="https://miro.medium.com/max/700/1*iLzCc-kwmxNVklZjqoCn_A.png">

</div>

</div>

---
handle: 'mariagrandury'
---

# MLOps workflow

<br>

MLOps Tools: MLflow, Airflow, Neptune, Kubeflow, MLrun...
- Add one step before deployment!

<br>

<div grid="~ cols-1" class="place-items-center">
<img style="height: 150px; margin-left: 0px" class="rounded" src="trustworthy_enough.png">
</div>

---
layout: center
logoHeader: 'https://mlopsworld.com/wp-content/uploads/2020/04/MLOps-Logo-2-150x150.png'
website: 'mariagrandury.github.io'
---

<div class="place-items-center">

# Thank you!

<br>

## Let's shape the future of AI Quality!
[aidkit.ai]()

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

# Code: Cleverhans

https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/tf2/mnist_tutorial.py

```py
# Train model with adversarial training
for epoch in range(FLAGS.nb_epochs):
  # keras like display of progress
  progress_bar_train = tf.keras.utils.Progbar(60000)
  for (x, y) in data.train:
      if FLAGS.adv_train:
          # Replace clean example with adversarial example for adversarial training
          x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
      train_step(x, y)

# Evaluate on clean and adversarial data
progress_bar_test = tf.keras.utils.Progbar(10000)
for x, y in data.test:
  y_pred = model(x)
  test_acc_clean(y, y_pred)

  x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
  y_pred_fgm = model(x_fgm)
  test_acc_fgsm(y, y_pred_fgm)

  x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
  y_pred_pgd = model(x_pgd)
  test_acc_pgd(y, y_pred_pgd)
```

---
handle: 'mariagrandury'
---

# Code: TextAttack

```sh
#!/bin/bash
# how to attack a DistilBERT model fine-tuned on SST2 dataset *from the
# huggingface model hub using the DeepWordBug recipe and 10 examples

textattack attack
  --model-from-huggingface distilbert-base-uncased-finetuned-sst-2-english
  --dataset-from-huggingface glue^sst2
  --recipe deepwordbug
  --num-examples 10
```

---
handle: 'mariagrandury'
---

# SHAP Explained

The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction.

The SHAP explanation method computes Shapley values from coalitional game theory. The feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly distribute the "payout" (= the prediction) among the features.

A player can be an individual feature value, e.g. for tabular data. A player can also be a group of feature values. For example to explain an image, pixels can be grouped to super pixels and the prediction distributed among them.

One innovation that SHAP brings to the table is that the Shapley value explanation is represented as an additive feature attribution method, a linear model. That view connects LIME and Shapley Values. SHAP specifies the explanation as: 

$$
g(z') = {\phi}_0+ \sum_{j=1}^{M} \phi_j {z′}_j 
$$

where $g$ is the explanation model,  
$ z′ \in \{0, 1\}^{M} $ is the coalition vector,
$M$ is the maximum coalition size and  
$ {\phi}_j \in \mathbb{R} $ is the feature attribution for a feature $j$, the Shapley values.

<!--
What I call "coalition vector" is called "simplified features" in the SHAP paper. I think this name was chosen, because for e.g. image data, the images are not represented on the pixel level, but aggregated to super pixels. I believe it is helpful to think about the z's as describing coalitions: In the coalition vector, an entry of 1 means that the corresponding feature value is "present" and 0 that it is "absent". This should sound familiar to you if you know about Shapley values. To compute Shapley values, we simulate that only some features values are playing ("present") and some are not ("absent"). The representation as a linear model of coalitions is a trick for the computation of the ϕ's. For x, the instance of interest, the coalition vector x' is a vector of all 1's, i.e., all feature values are "present". 
-->

[More about SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)

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
