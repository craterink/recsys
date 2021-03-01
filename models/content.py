import pytorch_lightning as pl
import torch

class ItemContentWeightedAverage(pl.LightningModule):
    def __init__(self, num_users, num_items, num_genres, thresh=0.5):
        super().__init__()
        # instantiate user profiles
        self.user_profiles = torch.zeros((num_users, num_genres))
        self.users_number_rated = torch.zeros((num_users,))

        # instantiate comparison functionality
        self.sim = torch.nn.CosineSimilarity()
        self.thresh = thresh
        
        # loss 
        self.Loss = torch.nn.BCELoss()

    def forward(self, x, item_features):
        # predict using current user profiles
        prob = self.sim(self.user_profiles[x[:,0]], item_features)
        predictions = prob > self.thresh

        # update user profiles
        self.user_profiles[x[:,0]] = (self.user_profiles[x[:, 0]]*self.users_number_rated[x[:, 0]] + item_features) / (self.users_number_rated[x[:, 0]] + 1)
        self.users_number_rated[x[:,0]] += 1

        return predictions


    def training_step(self, batch, batch_idx):
        x, item_features = batch
        preds = self(x, item_features)
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, item_features = batch
        preds = self(x, item_features)
        targets = x[:, 2].unsqueeze(-1).float()
        loss = self.Loss(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return None