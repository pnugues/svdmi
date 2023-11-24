# svdmi
These programs compute embeddings with a singular value decomposition (SVD or PCA). They use a cooccurrence matrix with mutual information (also called pointwise mutual information, PMI).

To run the program, collect a corpus of texts and place the texts in a folder. All the texts must have a `.txt` suffix, Then just run:
```
python cooc_mi_matrix.py folder_name
```
The embeddings will be saved in a `pca_vectors.txt` file.
