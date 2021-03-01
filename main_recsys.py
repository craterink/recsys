from fire import Fire
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from multiprocessing import Pool
from itertools import repeat

from movie_lens_data import *
from recsys_models import *

def main(udata_file, batch_size=100, neg_per_pos=4, name='run', run_big=False, perc_of_dataset=1, load_genres_data_file=''):
    run = wandb.init(project="recsys", reinit=True)
    run.name = '{}_neg_per_pos={}'.format(name, neg_per_pos)
    wandb_logger = WandbLogger()

    dl_kwargs = {
        'batch_size' : batch_size,
        'num_workers' : 4
    }
    if not len(load_genres_data_file): load_genres_data_file = None  
    if run_big:
        train_dataset = MovieLensImplicitDataset(udata_file, split='train', neg_per_pos=neg_per_pos, keep_perc_users=perc_of_dataset, using_20m=True, load_item_data_file=load_genres_data_file)
        val_dataset = MovieLensImplicitDataset(udata_file, split='val', neg_per_pos=neg_per_pos, keep_perc_users=perc_of_dataset, using_20m=True, load_item_data_file=load_genres_data_file)
    else:
        train_dataset = MovieLensImplicitDataset(udata_file, split='train', neg_per_pos=neg_per_pos, load_item_data_file=load_genres_data_file)
        val_dataset = MovieLensImplicitDataset(udata_file, split='val', neg_per_pos=neg_per_pos, load_item_data_file=load_genres_data_file)
    train_dl = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_dataset, shuffle=True, **dl_kwargs)
    num_users, num_items, user_idx_lookup, item_idx_lookup = train_dataset.num_users, train_dataset.num_items, train_dataset.user_idx_lookup, train_dataset.item_idx_lookup
    if load_genres_data_file:
        num_genres = train_dataset.item_genres.shape[1] - 1
        model = NeuralCollaborativeFilterWithItemGenres(num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup)
    else:
        model = NeuralCollaborativeFilter(num_users, num_items, user_idx_lookup, item_idx_lookup)
       
    trainer = pl.Trainer(automatic_optimization=True, logger=wandb_logger, callbacks=[EarlyStopping(monitor='val_loss')])
    # trainer = pl.Trainer(automatic_optimization=True, callbacks=[EarlyStopping(monitor='val_loss')])

    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    Fire(main)