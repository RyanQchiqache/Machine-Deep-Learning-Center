# Machine/Deep Learning Learning Center 

Welcome to the Machine/Deep Learning Learning Center! This repository is my personal space for exploring and understanding machine learning and deep learning concepts. It’s a growing collection of projects, experiments, and resources that reflect my journey as a student in this exciting field.

## Overview

As someone still learning, my goal is to document my progress, share my experiences, and provide clear examples, code snippets, and Jupyter notebooks. This repository covers topics ranging from foundational principles to advanced neural network techniques. While I’m still figuring things out and may make mistakes, I hope this can also be a helpful resource for others on a similar path.

##  Repository Structure

1. **cnn/**: Dive into Convolutional Neural Networks (CNNs) with detailed examples and implementations.
   - Includes logs, trained models (`best_model.h5`, `model.h5`), and the Python script `DigitRecognizer.py`.

2. **computerIntelligence/**: Explore artificial intelligence concepts and algorithms.
   - Contains the `policy_agent` folder with subdirectories such as `rooms/` (`rooms.py`, `__init__.py`), along with scripts like `agent.py`, `main.py`, and `N_Queen_SA.py`.

3. **docs/**: Quick guides for essential commands and file manipulations.
   - Includes `commands_help` folder with `commands_guide.html` and `python_file_manipulation.html`.

4. **gan/DCGAN/**: Uncover the power of Deep Convolutional Generative Adversarial Networks (DCGANs) with code and comprehensive explanations.
   - Includes subdirectories like `checkpoints/`, `dataset/`, `img/`, and `logs/`, and scripts like `Model_gen.py` and `Train_gen.py`.
   - **Paper:** [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

5. **learningML/**: Explore a broad spectrum of basic to intermediate machine learning concepts and techniques.
   - Includes subdirectories like `data/` (e.g., `ada_lovelace.txt`, `diabetes.csv`), `first_ml_models/` (e.g., KNN implementations), and `nlp_programming/` (e.g., text preprocessing).
   - Also contains model files (`fasttext.model`, `skipgram.model`) and experiment notes (`MyNoteBook.ipynb`).

6. **perceptron/**: Understand the fundamentals of the perceptron algorithm, a cornerstone of neural networks.
   - Includes implementations like `perceptron.py` and testing scripts.

7. **.gitignore**: Lists files and directories to be ignored by git.

8. **LICENSE**: The license governing the use of this repository.

9. **README.md**: The main README file you are currently reading.

##  Getting Started

1. **Clone the repository**:
    ```sh
    git clone https://github.com/RyanQchiqache/Machine-Learning-Learning-Center.git
    ```

2. **Navigate to the project directory**:
    ```sh
    cd Machine-Learning-Learning-Center
    ```

3. **Set up a virtual environment** (recommended):
    ```sh
    python -m venv my_env
    ```
   - **Activate the virtual environment**:
     - On Windows:
       ```sh
       .\my_env\Scripts\activate
       ```
     - On macOS/Linux:
       ```sh
       source my_env/bin/activate
       ```
   - **Install required libraries**:
     ```sh
     pip install numpy pandas matplotlib scipy scikit-learn tensorflow torch jupyter nltk seaborn
     ```
   - **Set up NLTK** (if required for natural language processing tasks):
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('wordnet')
     nltk.download('stopwords')
     nltk.download('averaged_perceptron_tagger')
     nltk.download('maxent_ne_chunker')
     nltk.download('words')
     ```
   - **Deactivate the virtual environment** when done:
     ```sh
     deactivate
     ```

4. **Alternatively, install required libraries globally** (if not using a virtual environment):
    - Using pip:
      ```sh
      pip install numpy pandas matplotlib scipy scikit-learn tensorflow torch jupyter nltk seaborn
      ```
    - Using conda:
      ```sh
      conda install numpy pandas matplotlib scipy scikit-learn tensorflow pytorch jupyter nltk seaborn
      ```

5. **Run Jupyter Notebooks**:
    Ensure Jupyter is installed and run:
    ```sh
    jupyter notebook
    ```

##  Content Details

### Convolutional Neural Networks (CNNs)
- Dive into the fundamentals and advanced concepts of CNNs.
- Explore projects with detailed logs, trained models (`best_model.h5`, `model.h5`), and practical implementations.
- Includes examples like `DigitRecognizer.py` to demonstrate CNN applications in image classification.

### Deep Convolutional Generative Adversarial Networks (DCGANs)
- Learn about Generative Adversarial Networks (GANs) and their innovative applications.
- Detailed walkthroughs of the DCGAN architecture and its components.
- Includes subdirectories such as `checkpoints/`, `dataset/`, and `img/` for practical demonstrations.
- Code examples like `Model_gen.py` and `Train_gen.py` to train and evaluate DCGAN models.

### Learning Machine Learning (learningML)
- A comprehensive collection of notebooks and scripts covering various machine learning algorithms.
- Topics include:
  - **Regression**: Linear and logistic regression implementations.
  - **Classification**: K-Nearest Neighbors (KNN), Support Vector Machines (SVMs), and more.
  - **Clustering**: Unsupervised learning techniques such as K-Means.
  - **Natural Language Processing (NLP)**: Preprocessing scripts, word embeddings (e.g., `fasttext.model`, `word2vec.model`), and semantic similarity analysis.
- Hands-on examples like `Evaluation_of_NN.ipynb` and `knn_from_scratch.py` to solidify understanding.

### Perceptron
- Introduction to the perceptron algorithm, a cornerstone of neural networks.
- Covers both the mathematical foundations and practical implementations.
- Scripts like `perceptron.py` and `perceptron_test.py` provide step-by-step guidance for building and testing perceptrons.

### Computational Intelligence and Search Algorithms
- Explore broader AI topics, such as policy agents and search algorithms.
- Includes examples like `N_Queen_SA.py` for solving the N-Queens problem using simulated annealing.
- The `policy_agent/` directory provides an in-depth look at agent-based models with a focus on intelligent decision-making.

### Documentation and Command Guides
- Quick reference materials in the `docs/` directory.
- Includes guides such as `commands_guide.html` and `python_file_manipulation.html` are html files for me to be able to have some of the code snipet or small block of code that i could forget with time, that would make my life easier.


##  Contribution

We welcome contributions to the Machine Learning Learning Center! If you have improvements, additional examples, or new topics you would like to add, please fork the repository and submit a pull request. Contributions can include:
- New machine learning algorithms or techniques.
- Enhancements to existing code and documentation.
- Additional explanations and tutorials to aid learning.

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

## ℹ About

This repository is maintained by Ryan Qchiqache and is intended for learning and educational purposes. It aims to provide valuable resources for anyone interested in learning machine learning, from beginners to advanced practitioners.
