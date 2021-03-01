import pytorch_lightning as pl
import torch

class NeuralCollaborativeFilter(pl.LightningModule):
    def __init__(self, num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup, user_embedding_size=8, item_embedding_size=8):
        super().__init__()
        # embedding layers
        self.user_idx_lookup = torch.tensor(user_idx_lookup).long()
        self.item_idx_lookup = torch.tensor(item_idx_lookup).long()
        self.uemb = torch.nn.Embedding(num_users, user_embedding_size)
        self.iemb = torch.nn.Embedding(num_items, item_embedding_size)

        # fully-connected layers
        fcinp_dim = user_embedding_size + item_embedding_size
        self.fc1 = torch.nn.Linear(fcinp_dim, fcinp_dim*4)
        self.fc2 = torch.nn.Linear(fcinp_dim*4, fcinp_dim*2)
        self.output = torch.nn.Linear(fcinp_dim*2, 1)
        self.FCNonLinear = torch.nn.ReLU()
        self.OutputNonLinear = torch.nn.Sigmoid()

        # loss 
        self.Loss = torch.nn.BCELoss()

    def forward(self, x, genres):
        # ignores genres
        users, items = self.user_idx_lookup[x[:, 0]], self.item_idx_lookup[x[:, 1]]
        emb = torch.cat([self.uemb(users), self.iemb(items)], dim=1)
        pred = self.OutputNonLinear(self.output(self.FCNonLinear(self.fc2(self.FCNonLinear(self.fc1(emb))))))
        return pred

    def training_step(self, batch, batch_idx):
        x, genres = batch
        preds = self(x, genres)
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, genres = batch
        preds = self(x, genres)
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class NeuralCollaborativeFilterWithItemGenres(pl.LightningModule):
    def __init__(self, num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup, user_embedding_size=8, item_embedding_size=8, genres_embedding_size=4):
        super().__init__()
        # embedding layers
        self.user_idx_lookup = torch.tensor(user_idx_lookup).long()
        self.item_idx_lookup = torch.tensor(item_idx_lookup).long()
        self.uemb = torch.nn.Embedding(num_users, user_embedding_size)
        self.iemb = torch.nn.Embedding(num_items, item_embedding_size)
        self.gemb = torch.nn.Linear(num_genres, genres_embedding_size)

        # fully-connected layers
        fcinp_dim = user_embedding_size + item_embedding_size + genres_embedding_size
        self.fc1 = torch.nn.Linear(fcinp_dim, fcinp_dim*4)
        self.fc2 = torch.nn.Linear(fcinp_dim*4, fcinp_dim*2)
        self.output = torch.nn.Linear(fcinp_dim*2, 1)
        self.FCNonLinear = torch.nn.ReLU()
        self.OutputNonLinear = torch.nn.Sigmoid()

        # loss 
        self.Loss = torch.nn.BCELoss()

    def forward(self, x, item_genres):
        users, items = self.user_idx_lookup[x[:, 0]], self.item_idx_lookup[x[:, 1]]
        emb = torch.cat([self.uemb(users), self.iemb(items), self.gemb(item_genres)], dim=1)
        pred = self.OutputNonLinear(self.output(self.FCNonLinear(self.fc2(self.FCNonLinear(self.fc1(emb))))))
        return pred

    def training_step(self, batch, batch_idx):
        x, genres = batch
        preds = self(x, genres.float())
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, genres = batch
        preds = self(x, genres.float())
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())