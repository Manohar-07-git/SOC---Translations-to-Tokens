# 🚀 Translations to Tokens

[![Deep Learning](https://img.shields.io/badge/Topic-Deep%20Learning-blue.svg)](https://github.com/Manohar-07-git)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-red.svg)](https://jupyter.org/)

A hands‑on repository featuring curated Jupyter notebooks and datasets designed for exploring introductory deep learning and sequence models. This project walks through building fundamental architectures from scratch—moving from micrograd autodiff engines to character‑level language models, custom Multi-Layer Perceptrons (MLPs), and WaveNet-style implementations.

---

## 📅 Project Workflow & Timeline

Here is the step-by-step breakdown of concepts covered, models built, and milestones achieved week-by-week.

### 🔹 Week 1: Foundations & The Bigram Model

#### Lecture 1: Micrograd from Scratch
* Built a custom **micrograd** automatic differentiation engine entirely from scratch.
* Visualized computation graphs dynamically using `Graphviz` (`Digraph`).
* Designed intuitive, low-level object-oriented implementations for neural network components by building custom `Neuron`, `Layer`, and `MLP` classes.

<p align="center">
  <img src="https://camo.githubusercontent.com/9e557511ab3e3b3d28f7c8c7cd881d8ccef22c12161783eb65823861aa21a47f/68747470733a2f2f63733233316e2e6769746875622e696f2f6173736574732f6e6e312f6e6575726f6e5f6d6f64656c2e6a706567" alt="Neuron Model" width="500">
</p>

#### Lecture 2: Character-Level Bigram Language Model
1. **The Counting Method:** Processed a text dataset to count character-to-character transitions (bigrams). Implemented standard string-to-integer (`stoi`) and integer-to-string (`itos`) mapping functions to construct a clean frequency lookup matrix.
   
   <p align="center">
     <img src="./num.jpeg" alt="Bigram Count Matrix" width="500">
   </p>

2. **The Neural Network Method:** Built a single-layer neural network proxy for the bigram model. 
   * Formulated the dataset pipeline using **One-Hot Encoding**.
   * Initialized random weights and biases.
   * Hand-wrote the cross-entropy loss function ($Matrix\ Multiplication \rightarrow Exponentiation \rightarrow Softmax\ Probabilities$ using PyTorch broadcasting).
   * Optimized parameters using manual gradient descent.

---

### 🔹 Week 2: Embeddings, Initialization & Normalization

#### Lecture 3: MLP & Character Embeddings
* Implemented a Multi-Layer Percepton based on the seminal [Bengio et al. 2003 Neural Language Model Paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
* Developed distributed **Character Embeddings**. Projecting tokens into a dense continuous space allows the network to automatically cluster semantic similarities (e.g., teaching the model that `'A'` and `'The'` share similar syntactic profiles).
* Studied techniques for diagnosing and choosing an optimal **Learning Rate (LR)**.

<p align="center">
  <img src="https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTG5XddhcG7VHy0ukdk93ZqrpM8RVPZFMBtcpI3agM2Dm5XnU92686AY9c&s=10" alt="Embedding Space" width="400">
</p>

#### Lecture 4: Activations & Deep Network Dynamics
* Explored deep-network stabilization inspired by the [Kaiming He et al. Paper](https://arxiv.org/pdf/1502.01852) and the [Batch Normalization Paper](https://arxiv.org/pdf/1502.03167).
* **The Math Behind Kaiming Init:** Xavier initialization fails with ReLU because negative values get zeroed out ($f(x) = \max(0, x)$), cutting variance in half at every layer and leading to vanishing signals. Kaiming stabilizes deep networks by dynamically scaling weights using the input dimensionality (`fan_in`):
  
  $$\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$

#### 📝 Core Assignments Completed:
* Trained a model on the **MNIST Handwritten Digit Dataset**.
* Practiced **Layer Folding** techniques post-training for inference optimization.
* Systematically reduced validation loss baselines established in previous lectures.

---

### 🔹 Week 3: Framework Internals & WaveNet Architectures

#### Framework Familiarity (PyTorch Scratchpad)
* Developed custom, manual implementations of standard PyTorch structural components including `BatchNorm1d` and `Linear` layers to understand under-the-hood tensor manipulations.

#### Lecture 5: Manual Backpropagation Mastery
* Deep-dive into writing raw backpropagation algorithms by hand (recreating the classic 2012–2015 production pipeline workflow). Moved from basic chain-rule derivations to professional-level gradient tracking across multi-layer graph structures.

#### Lecture 6: Hierarchical Sequence Models (WaveNet Style)
* Implemented a dilated, tree-like sequence structure inspired by DeepMind's [WaveNet Architecture Paper](https://arxiv.org/pdf/1609.03499).
* Instead of squeezing an entire mini-batch context length flat immediately, the model groups consecutive byte/token pairs hierarchically across progressive layers, dramatically optimizing context efficiency and improving downstream loss metrics.