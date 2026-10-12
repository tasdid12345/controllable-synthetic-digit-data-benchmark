# Class-conditional filtered synthetic data vs. standard augmentation on digits

## Motivation
Labeled document images are expensive, and synthetic generation is a common way to stretch a small
annotation budget. The DocDjinn paper, "DocDjinn: controllable synthetic document generation with
VLMs and handwriting diffusion" (2026), proposes generating controllable synthetic documents with
vision-language models and handwriting diffusion. This experiment asks a narrower question using a
simpler, fully reproducible stand-in.

## Question
Does a class-conditional, quality-filtered synthetic generator give a downstream classifier more test
accuracy than standard image augmentation when only a few real labeled examples per class are available?

## Dataset
scikit-learn's bundled `load_digits` (1,797 grayscale 8x8 character images, no download). It is used as
a stand-in for character crops from document pages. It is not the paper's own dataset. A stratified
70/30 train/test split is used, and 15 real training images per class are kept.

## Baseline vs. improved
- **Baseline:** a standard augmentation generator (random shifts, small rotations, pixel noise) produces
  10 synthetic images per real image. An MLPClassifier is trained on real plus synthetic images.
- **Improved:** a class-conditional Gaussian generator in PCA space, with shrinkage covariance, is fitted
  per class on the same real images. Samples are kept only if a logistic regression trained on the real
  images assigns them to the intended class with probability at least 0.8. The same MLPClassifier is
  trained on real plus kept samples.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a bar chart to `figures/accuracy_bar.png`).

## Reference

This project is motivated by the following paper:

> M Lamott, S Saifullah, N Riaz, YN Weweler (2026). *DocDjinn: controllable synthetic document generation with VLMs and handwriting diffusion*. International Journal on. https://link.springer.com/article/10.1007/s10032-026-00604-7

Paper author: Prof. Andreas Dengel, RPTU Kaiserslautern-Landau (andreas.dengel@dfki.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
