from fire import Fire
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from multiprocessing import Pool
from itertools import repeat

from data.movielens100k import *
from models.content import *
from models.ncf import *

def main(udata_file, genres_file, batch_size=100, neg_per_pos=4, wandb_name='', emb_size=8, model_type='ncf', tf_idf=False):
    # init WANDB
    if wandb_name:
        args = locals()
        del args['udata_file']
        del args['genres_file']
        del args['wandb_name']
        run = wandb.init(project=wandb_name, reinit=True)
        run.name = str(args)
        wandb_logger = WandbLogger()

    # init MODELS Metadata
    Models = {
        'ncf' : (NeuralCollaborativeFilter, {}, {
            'user_embedding_size': emb_size,
            'item_embedding_size': emb_size
        }, {}),
        'ncf-g' : (NeuralCollaborativeFilterWithItemGenres, {}, {
            'user_embedding_size': emb_size,
            'item_embedding_size': emb_size,
            'genre_embedding_size' : 4
        }, {}),
        'content-i-wavg' : (ItemContentWeightedAverage, {
            'tf_idf' : tf_idf
        },
        {
            'thresh' : 0.5
        }, {
            'automatic_optimization' : False
        }),
    }
    Model, DatasetKwargs, ModelKwargs, ModelTrainerKwargs = Models[model_type]

    # init DATA
    dl_kwargs = {
        'batch_size' : batch_size,
        'num_workers' : 4
    }
    train_dataset = MovieLens100KWithGenresDataset(udata_file, genres_file, split='train', neg_per_pos=neg_per_pos, **DatasetKwargs)
    val_dataset = MovieLens100KWithGenresDataset(udata_file, genres_file, split='val', neg_per_pos=neg_per_pos, **DatasetKwargs)
    train_dl = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_dataset, shuffle=True, **dl_kwargs)
    num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup = train_dataset.get_user_item_info()

    # init MODEL
    model = Model(num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup, **ModelKwargs)       

    # TRAIN
    TrainerKwargs = {
        'automatic_optimization' : True
    }
    TrainerKwargs.update(ModelTrainerKwargs)
    trainer = pl.Trainer(logger=(wandb_logger if wandb_name else None), callbacks=[EarlyStopping(monitor='val_loss')], **TrainerKwargs)
    trainer.fit(model, train_dl, val_dl)

    # TEST / EVALUATE
    # TODO

if __name__ == "__main__":
    Fire(main)