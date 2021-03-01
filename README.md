## Learning about Recommendation Systems

This repo contains the work I've done to learn more about recommendation systems. I am working with the MovieLens 100K dataset and predicting user-item interactions. The structure of the repo is as follows:
* `models/`: contains models used to learn unknown user-item ratings
* `data/`: contains code / torch Datasets to load historical ratings
* `main_*.py`: training code

The models I have worked with so far include:
* *Item Content-based Recommendation*: learn a user's genre preferences, and predict based on the similarity between a user's historical genre preferences and a given movie genre.
* *Neural Collaborative Filter*: learn user and item embeddings and how to combine them for prediction.
* *Neural Collaborative Filter with Item Features (Movie Genres)*: learn user embeddings, item embeddings, and genre embeddings and how to combine them for prediction.

## Running this Code
To run the code, you must have several dependencies installed, including PyTorch and PyTorch Lightning: `pip install -r requirements.txt`. You must also download the MovieLens 100K dataset, possibly from [here](https://www.kaggle.com/prajitdatta/movielens-100k-dataset).

Run a given model using its prefix in the Models dict: e.g., `python main_recsys100k.py path/to/u.data path/to/u.item --model_type ncf-g [kwargs]`. Models:
```
Models = {
        'ncf' : (NeuralCollaborativeFilter, {
            'user_embedding_size': emb_size,
            'item_embedding_size': emb_size
        }),
        'ncf-g' : (NeuralCollaborativeFilterWithItemGenres, {
            'user_embedding_size': emb_size,
            'item_embedding_size': emb_size,
            'genre_embedding_size' : 4
        }),
        'content-i-wavg' : (ItemContentWeightedAverage, {
            'thresh' : 0.5
        }),
    }
```
