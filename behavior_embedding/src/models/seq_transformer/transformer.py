# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
import csv
import importlib
import math
import random
from datetime import datetime

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, embed_list=None, attention_list=None):
        # Attention part
        if attention_list is None:
            attn_out = self.self_attn(x, mask=mask)
        else:
            attn_out, attention = self.self_attn(x, mask=mask, return_attention=True)
            attention_list.append(attention)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        if embed_list is not None:
            embed_x = torch.squeeze(x.detach().cpu())
            embed_x = torch.swapaxes(embed_x, -1, 0)
            embed_list.append(embed_x.numpy())
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None, embed_list=None, attention_list=None):
        for i_layer, layer in enumerate(self.layers):
            if i_layer == len(self.layers) - 1:
                x = layer(
                    x, mask=mask, embed_list=embed_list, attention_list=attention_list
                )
            else:
                x = layer(x, mask=mask, attention_list=attention_list)
        return x


class Encoding(nn.Module):
    def __init__(self, d_model, device, seq_length, mask_pos):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

        self.mask_embed = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        self.device = device
        self.mask_pos = mask_pos
        self.seq_length = seq_length

    def forward(self, x):
        mask_idx = np.zeros(self.seq_length, dtype=np.int32)

        for pos in self.mask_pos:
            mask_idx[pos] = 1

        mask_idx = torch.from_numpy(mask_idx).cuda()

        pos_emb = self.pe[:, : x.size(1)]
        x = x + pos_emb
        x = x + self.mask_embed(mask_idx).unsqueeze(0)
        return x


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerPredictor(L.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        image_encoder_module,
        image_encoder_class,
        encoder_checkpoint,
        dataset_mean,
        dataset_std,
        seq_length,
        mask_pos,
        mask_perc,
        dropout=0.0,
        input_dropout=0.0,
        random_masking=False,
        random_masking_per_forward=True,
        loss_type="MSE",
        imp_factor=1,
        enc_rotation: bool = True,
    ):
        """TransformerPredictor.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        """
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.image_encoder_module = image_encoder_module
        self.image_encoder_class = image_encoder_class
        self.encoder_checkpoint = encoder_checkpoint

        self.dec_imgs = None
        self.dec_imgs_inp = None
        self.decoded = False

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.seq_length = seq_length
        self.mask_pos = mask_pos
        self.mask_perc = mask_perc
        self.random_masking = random_masking
        self.random_masking_per_forward = random_masking_per_forward
        self.embed_list = []
        self.attention_list = []
        self.collected_pred_rotations = []
        self.collected_label_rotations = []
        self.loss_type = loss_type
        self.imp_factor = imp_factor

        self.enc_rotation = enc_rotation

        # if rotation encoding is turned on, the model embedding dim will be +1 because of the rotation info added
        if self.enc_rotation:
            self.input_dim += 1

        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.image_encoder_model = importlib.import_module(self.image_encoder_module)
        self.image_encoder_class = getattr(
            self.image_encoder_model, self.image_encoder_class
        )

        self.image_encoder = self.image_encoder_class.load_from_checkpoint(
            self.encoder_checkpoint,
            strict=False,
            input_dim=(self.input_dim if not self.enc_rotation else self.input_dim - 1),
        )

        self.image_encoder.freeze()
        self.image_encoder.eval()

        if self.random_masking:
            self.num_masks = round(self.mask_perc * self.seq_length / 100)
            self.mask_pos = random.sample(range(self.seq_length), self.num_masks)

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout), nn.Linear(self.input_dim, self.model_dim)
        )
        # Positional encoding for sequences
        self.encoding = Encoding(
            self.model_dim, self.device, self.seq_length, self.mask_pos
        )
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.input_dim),
        )

        # create diagonal attention mask with mask = 0
        diag_mask = ~torch.eye(self.seq_length, dtype=torch.bool).cuda()
        self.att_layer_mask = diag_mask.type(torch.float32)

    def plot_attention_maps(self, input_data, attn_maps, idx=0):
        if input_data is not None:
            input_data = input_data[idx].detach().cpu().numpy()
        else:
            input_data = np.arange(attn_maps[idx].shape[-1])

        # get batch
        attn_maps = attn_maps[idx].detach().cpu().numpy()

        num_heads = attn_maps[0].shape[0]
        num_layers = len(attn_maps)
        seq_len = input_data.shape[0]
        fig_size = 4 if num_heads == 1 else 3
        fig, ax = plt.subplots(
            num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size)
        )
        if num_layers == 1:
            ax = [ax]
        if num_heads == 1:
            ax = [[a] for a in ax]
        for row in range(num_layers):
            for column in range(num_heads):
                ax[row][column].imshow(attn_maps[row][column], origin="lower", vmin=0)
                ax[row][column].set_xticks(list(range(seq_len)))
                ax[row][column].set_xticklabels(input_data.tolist())
                ax[row][column].set_yticks(list(range(seq_len)))
                ax[row][column].set_yticklabels(input_data.tolist())
                ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
        fig.subplots_adjust(hspace=0.5)
        self.logger.experiment.log({"Test/Attention": fig})

    def forward(
        self,
        x,
        rotation,
        mask=None,
        add_positional_encoding=True,
        store_latents=False,
        get_attention=False,
    ):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """

        # get image embeddings
        embeddings = self.image_encoder(x)
        if rotation is not None:
            embeddings = torch.cat([embeddings, rotation[..., None]], dim=-1)

        # generate new random masks positions each forward pass
        if self.random_masking and self.random_masking_per_forward:
            self.mask_pos = random.sample(range(self.seq_length), self.num_masks)

        # get gt and mask images
        y = embeddings.detach().clone()
        for pos in self.mask_pos:
            embeddings[:, pos, :] = torch.zeros_like(embeddings[:, pos, :])

        x = embeddings.detach().clone()
        x = self.input_net(x)

        if add_positional_encoding:
            x = self.encoding(x)

        embed_list = None
        if store_latents:
            embed_list = self.embed_list

        attention_list = None
        if get_attention:
            attention_list = self.attention_list

        x = self.transformer(
            x, mask=mask, embed_list=embed_list, attention_list=attention_list
        )
        x = self.output_net(x)
        return x, y

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.warmup,
            max_iters=self.trainer.estimated_stepping_batches,  # self.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        if self.enc_rotation:
            inp_data, rotation = batch
        else:
            inp_data = batch
            rotation = None

        # Perform prediction and calculate loss and accuracy
        store_latents = False
        get_attention = False
        if mode == "test":
            store_latents = True
            get_attention = True

        preds, labels = self.forward(
            inp_data,
            rotation,
            mask=self.att_layer_mask,
            add_positional_encoding=True,
            store_latents=store_latents,
            get_attention=get_attention,
        )

        if self.enc_rotation:
            pred_rotations, label_rotations = preds[..., -1], labels[..., -1]
            preds, labels = preds[..., :-1], labels[..., :-1]

        if (
            mode == "val"
            and not self.decoded
            and hasattr(self.image_encoder, "decoder")
        ):
            self.dec_imgs_inp = torch.squeeze(inp_data[:, self.mask_pos[-1], :, :, :])
            self.dec_imgs = self.image_encoder.decoder(
                torch.squeeze(preds[:, self.mask_pos[-1], :])
            )
            self.decoded = True

        # set loss function type
        if self.loss_type == "MSE":
            loss_func = F.mse_loss
        elif self.loss_type == "MAE":
            loss_func = F.l1_loss
        else:
            NotImplementedError(f"Loss function {self.loss_type} not implemented yet.")

        # imputation loss
        imp_loss = loss_func(preds[:, self.mask_pos, :], labels[:, self.mask_pos, :])

        # get not masked indices
        non_masked_list = list(range(0, self.seq_length))
        for m in self.mask_pos:
            non_masked_list.remove(m)

        # rotation loss
        if self.enc_rotation:
            rotation_loss = loss_func(
                pred_rotations[:, self.mask_pos], label_rotations[:, self.mask_pos]
            )

        # total loss
        loss = self.imp_factor * imp_loss
        if self.enc_rotation:
            loss += rotation_loss

        # Logging
        logs = {
            "Imp_Loss": imp_loss,
            "Loss": loss,
        }

        if self.enc_rotation:
            logs["Rotation_Loss"] = rotation_loss
            if mode == "test":
                self.collected_pred_rotations.append(
                    np.squeeze(pred_rotations[:, self.mask_pos].detach().cpu().numpy())
                )
                self.collected_label_rotations.append(
                    np.squeeze(label_rotations[:, self.mask_pos].detach().cpu().numpy())
                )

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._calculate_loss(batch, mode="train")
        self.log_dict(
            {f"Train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._calculate_loss(batch, mode="val")
        self.log_dict({f"Val/{k}": v for k, v in logs.items()})
        return loss

    def on_validation_epoch_start(self):
        self.dec_imgs_inp = None
        self.dec_imgs = None
        self.decoded = False

    def on_validation_epoch_end(self):
        if hasattr(self.image_encoder, "decoder"):
            imgs = (
                torch.stack([self.dec_imgs_inp, self.dec_imgs], dim=1)
                .flatten(0, 1)
                .cpu()
            )
            mean, std = np.array(self.dataset_mean), np.array(self.dataset_mean)
            grid = (
                torchvision.utils.make_grid(imgs, nrow=32).permute(1, 2, 0).numpy()
                * std
                + mean
            ) * 255
            self.logger.experiment.log(
                {"Val/Preds": [wandb.Image(torch.tensor(grid).permute(2, 0, 1))]}
            )

    def test_step(self, batch, batch_idx):
        loss, logs = self._calculate_loss(batch, mode="test")
        self.log_dict({f"Test/{k}": v for k, v in logs.items()})
        return loss

    def on_test_epoch_end(self):
        # store embeddings in wandb
        embeds = np.array(self.embed_list)
        embeds = np.mean(embeds, axis=-1)

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        with open(f"latents/{dt_string}_latents.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(embeds)

        # get attention maps
        att_maps = self.attention_list
        self.plot_attention_maps(input_data=None, attn_maps=att_maps)

        columns = [str(x) for x in self.mask_pos]
        tbl = wandb.Table(columns=columns, data=self.collected_pred_rotations)
        self.logger.experiment.log({"Test/Pred_Rotations": tbl})

        columns = [str(x) for x in self.mask_pos]
        tbl = wandb.Table(columns=columns, data=self.collected_label_rotations)
        self.logger.experiment.log({"Test/Label_Rotations": tbl})
