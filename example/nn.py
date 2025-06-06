import polars as pl
import lightning.pytorch as pyt
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F

from morphers import Normalizer, Integerizer



def morph_data(csv_path, cat_cols, num_cols, target_col = None, saved_morphers = None):
    """
    should include saved states
    """
    df = pl.read_csv(csv_path)
    
    if saved_morphers is None:
        cat_morphers = {
            feature: Integerizer.from_data(df[feature]) for feature in cat_cols
        }

        num_morphers = {
            feature: Normalizer.from_data(df[feature]) for feature in num_cols
        }

        morphers = cat_morphers | num_morphers
    else:
        morphers = saved_morphers

    if target_col is not None:
        df = df.filter(~pl.col(target_col).is_null())

        df = (
            df.select(
                target_col,
                # applies morphers
                *[
                    morpher(morpher.fill_missing(pl.col(feature))).alias(feature)
                    for feature, morpher in morphers.items()
                ]
            ).with_row_index()
        )
    else:
        df = (
            df.select(
                target_col,
                # applies morphers
                *[
                    morpher(morpher.fill_missing(pl.col(feature))).alias(feature)
                    for feature, morpher in morphers.items()
                ]
            ).with_row_index()
        )

    return df, morphers


class ThingDataset(Dataset):
    def __init__(
        self,
        df,
        morphers,
        target_col = None,
    ):
        super().__init__()
        self.df = df
        self.target_col = target_col
        self.morphers = morphers

    def __len__(self):
        return self.df.height


    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        inputs = {
            feat: torch.tensor(row[feat], dtype=morpher.required_dtype)
            for feat, morpher in self.morphers.items()
        } | {"index": row["index"]}
        
        if self.target_col is not None:
            inputs = inputs | {
                "target": torch.tensor(row[self.target_col])
            }

        return inputs


class ThingNet(nn.Module):
    def __init__(self, morphers, hidden_dim, out_cats):
        super().__init__()
        self.morphers = morphers
        
        self.embedders = nn.ModuleDict(
            {
                feat: morpher.make_embedding(hidden_dim)
                for feat, morpher in morphers.items()
            }
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

        self.classifier = nn.Linear(hidden_dim, out_cats)

    def forward(self, x):
        # n x k x e
        embeddings = torch.stack(
            [
                embed(x[feat])
                for feat, embed in self.embedders.items()
            ],
            dim=-2,
        )

        summed = embeddings.sum(dim=-2)
        summed = self.activation(self.norm(summed))

        return self.classifier(summed)
        

class ThingLightning(pyt.LightningModule):
    def __init__(
        self,
        morphers,
        target_col,
        hidden_dim,
        out_cats,
        optim_lr,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_lr = optim_lr
        
        self.global_log_step = 0

        self.net = ThingNet(
            morphers=morphers,
            hidden_dim=hidden_dim,
            out_cats=out_cats,
        )

        self.apply(self._init_weights)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.optim_lr)

    def training_step(self, x):
        preds = self.net(x)
        loss = {"train_loss": self.criterion(preds, x["target"])}
        self.global_log_step += x["target"].shape[0]
        
        self.log_dict(loss)
        return loss["train_loss"]

    def validation_step(self, x):
        preds = self.net(x)
        loss = {"valid_loss": self.criterion(preds, x["target"])}
        self.global_log_step += x["target"].shape[0]
        
        self.log_dict(loss)
        return loss["valid_loss"]

    def predict_step(self, x):
        return F.softmax(self.net(x), dim=-1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)