import pytorch_lightning as pl
import torch

class ItemContentWeightedAverage(pl.LightningModule):
    def __init__(self, num_users, num_items, num_genres, user_idx_lookup, item_idx_lookup, thresh=0.5):
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
        users_idxs = x[:, 0] - 1 # 1-indexed to 0-indexed
        predictions = self.sim(self.user_profiles[users_idxs], item_features).float()

        # update user profiles (only for positive interactions)
        pos_batch_idxs = x[:,2] == 1
        pos_user_idxs = users_idxs[pos_batch_idxs]
        self.user_profiles[pos_user_idxs] = ((self.user_profiles[pos_user_idxs]*self.users_number_rated[pos_user_idxs].unsqueeze(1) + item_features[pos_batch_idxs, :]) / (self.users_number_rated[pos_user_idxs].unsqueeze(1) + 1)).float()
        self.users_number_rated[pos_user_idxs] += 1

        return predictions


    def training_step(self, batch, batch_idx):
        x, item_features = batch
        preds = self(x, item_features)
        targets = x[:, 2].float()
        loss = self.Loss(preds, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return None

    def validation_step(self, batch, batch_idx):
        x, item_features = batch
        preds = self(x, item_features)
        targets = x[:, 2].float()
        loss = self.Loss(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return None

    def configure_optimizers(self):
        return None