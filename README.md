
# Spotify Genre Classification and Content-Based Recommendation

This repository contains the code and experiments for an advanced statistical learning project on genre-aware, content-based music recommendation using Spotify track features. All code is carried out in a Jupyter Notebook and includes annotations for clarification.

A tabular neural network (multi-layer perceptron with categorical embeddings) is trained to predict the genre of a track, and its penultimate layer is reused as a learned embedding for nearest-neighbour recommendation. The main experimental focus is on the effect of batch size and the interaction between model depth and learning rate.

---

## Project Summary

- Input: track-level audio and metadata features from a public Spotify dataset.
- Task: supervised multi-class classification of `track_genre`.
- Model: MLP for tabular data with embeddings for categorical variables.
- Recommender: cosine-similarity retrieval in the learned embedding space.
- Hyperparameters studied:
  - batch size (impact on speed and generalisation)
  - model depth × learning rate (capacity vs. optimisation)

The work was carried out in the context of an Advanced Statistical Learning seminar at HTW Berlin.

---

## Data

Source: Spotify Tracks Genre Dataset (Kaggle)  
https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset

Target label:

- `track_genre` (encoded to an integer label `genre_idx`)

Input features:

- Numeric: `popularity`, `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`,
  `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- Categorical: `explicit`, `key`, `mode`, `time_signature`

The genre label is used only as the target for training and evaluation and is never fed in as an input feature.

---

## Repository Layout

```text
.
├── Notebook (code)           
  └── StatisticalLearning_PO_SpotifyRecommender_final.ipynb   # main notebook with results and architecture
├── src                                                       # in progress currently
└── README.md
```

If the CSV is stored elsewhere, adjust the `DATA_PATH` variable in the notebook.

---

## Environment and Installation

Python version: 3.8 or later.

Main dependencies:

- numpy
- pandas
- scikit-learn
- matplotlib
- torch (PyTorch)
- jupyter or jupyterlab

Example setup:

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn matplotlib torch jupyter
# or: pip install -r requirements.txt
```

Then start Jupyter and open the main notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Run the cells from top to bottom.

---

## Model and Training

Preprocessing:

- Numeric features are standardised with `StandardScaler`, fitted on the training split only.
- Categorical features are mapped to integer IDs; unseen categories at test time go to a reserved ID.

Model:

- Multi-layer perceptron for tabular data:
  - embeddings for each categorical feature
  - concatenation of standardised numeric features and categorical embeddings
  - stacked `Linear → ReLU → Dropout` hidden layers
  - final linear layer producing logits over genres
- Penultimate layer output `h(x)` is used as a track embedding for the recommendation component.

Training setup:

- Loss: multi-class cross-entropy
- Optimiser: AdamW
- Split: 70 % train, 15 % validation, 15 % test (stratified by genre)
- Early stopping: monitor validation macro F1; reload the best checkpoint before testing.

Evaluation metrics:

- Classification: accuracy and macro F1 (unweighted average over all genres)
- Recommendation: Genre-Hit@10 (fraction of top-10 recommendations that share the query’s genre, averaged over many queries)

For a general introduction to neural networks and MLPs, see:

- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.

---

## Recommendation Component

After training the classifier, the model is reused as a feature extractor:

1. Compute the penultimate representation `h(x)` for all tracks.
2. L2-normalise embeddings: `h_norm(x) = h(x) / ||h(x)||`.
3. For a query track `q` and candidate track `i`, compute cosine similarity  
   `sim(q, i) = h_norm(q) · h_norm(i)`.
4. Rank candidate tracks by similarity and return the top-k nearest neighbours.

Because there are no user–item interactions in the dataset, recommendation quality is evaluated with Genre-Hit@10 as a label-based proxy: for each query, the metric measures how many of the top-10 neighbours share the query’s genre.

---

## Hyperparameter Experiments

All experiments use the same data split and preprocessing; only the chosen hyperparameters differ.

### Batch Size

Question: How does batch size affect training time and generalisation?

Setup:

- Hidden sizes: `[256, 256, 128]`
- Optimiser: AdamW, learning rate `1e-3`
- Batch sizes: `64`, `256`, `1024`, `4096`
- Up to 20 epochs with early stopping

Summary:

| Batch size | sec/epoch | Test accuracy | Macro F1 | Genre-Hit@10 |
|-----------:|----------:|--------------:|---------:|-------------:|
| 64         | 2.82      | 0.319         | 0.298    | 0.219        |
| 256        | 1.55      | 0.312         | 0.291    | 0.216        |
| 1024       | 1.17      | 0.299         | 0.274    | 0.204        |
| 4096       | 1.28      | 0.269         | 0.239    | 0.189        |

Observations:

- Larger batches speed up each epoch.
- Very large batches show reduced test accuracy and Genre-Hit@10, consistent with known large-batch generalisation issues.

Related figures: Figure 1–3.

### Depth and Learning Rate

Question: How do depth and learning rate interact with respect to performance and runtime?

Setup:

- Hidden width: 128 units per hidden layer
- Depths: 2, 6, 10
- Learning rates: `1e-4`, `3e-4`, `1e-3`, `3e-3`
- Batch size: 512
- Up to 25 epochs with early stopping

Example (depth = 2):

| Depth | Learning rate | sec/epoch | Test accuracy | Macro F1 | Genre-Hit@10 |
|------:|--------------:|----------:|--------------:|---------:|-------------:|
| 2     | 1e-4          | 1.03      | 0.222         | 0.187    | 0.169        |
| 2     | 3e-4          | 1.00      | 0.259         | 0.229    | 0.181        |
| 2     | 1e-3          | 0.97      | 0.299         | 0.276    | 0.192        |
| 2     | 3e-3          | 0.97      | 0.317         | 0.301    | 0.203        |

Observations:

- Learning rate strongly affects macro F1; very small values underfit.
- Increasing depth mainly increases runtime and does not clearly outperform the shallow model within the available training budget.
- For this dataset, a relatively shallow MLP with a well-tuned learning rate is sufficient.

Related figures: Figure 4–5.

---

## Reproducibility

To reproduce the experiments:

1. Download the Kaggle dataset and save it as `spotify_train.csv` in the repository root (or adjust the path in the notebook).
2. Create a Python environment and install the required packages.
3. Open `StatisticalLearning_PO_SpotifyRecommender.ipynb` (or the annotated version) and execute all cells in order.

The notebook sets a fixed random seed for reproducibility, although small numerical differences across machines and PyTorch versions are expected.

---

## References

- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.  
- Guo, C., & Berkhahn, F. (2016). Entity Embeddings of Categorical Variables. arXiv:1604.06737.  
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.  
- Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv:1711.05101.  
- Keskar, N. S., et al. (2016). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. arXiv:1609.04836.  
- Hoffer, E., Hubara, I., & Soudry, D. (2017). Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks. NeurIPS.

---

## License

This repository was created for academic use in the Advanced Statistical Learning seminar at HTW Berlin.  
If you intend to reuse the code, please add an appropriate open-source license (e.g. MIT) or contact the author.
