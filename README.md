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
To run the code, you must have several dependencies installed, including PyTorch and PyTorch Lightning: `pip install -r requirements.txt`. 
