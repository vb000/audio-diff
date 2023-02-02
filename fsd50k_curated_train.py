import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import wandb

class FSD50KCurated(Dataset):
    def __init__(self, root_dir, sr=44100, max_len=2**18):
        self.root_dir = root_dir
        self.sr = sr
        self.max_len = max_len

        # List sub-directories and dir names are labels
        self.labels = os.listdir(root_dir)

        # List all files in each sub-directory
        self.files = {}
        for label in self.labels:
            self.files[label] = os.listdir(os.path.join(root_dir, label))
        
        # Create a list of tuples (label, file)
        self.data = []
        for label in self.labels:
            for file in self.files[label]:
                self.data.append((label, file))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, file = self.data[idx]
        path = os.path.join(self.root_dir, label, file)
        audio, fs = torchaudio.load(path)
        if fs != self.sr:
            audio = torchaudio.functional.resample(audio, fs, self.sr)
        audio = audio[:, :self.max_len]
        if audio.shape[1] < self.max_len:
            audio = torch.nn.functional.pad(audio, (0, self.max_len - audio.shape[1]))
        return audio, label

def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = [audio.permute(1, 0) for audio in audios]
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    audios = audios.permute(0, 2, 1)
    return audios, list(labels)

class FSD50KDiffusionModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = DiffusionModel(
            net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
            in_channels=1, # U-Net: number of input/output (audio) channels
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
            attention_heads=8, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            diffusion_t=VDiffusion, # The diffusion method used
            sampler_t=VSampler, # The diffusion sampler used
            use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
            use_embedding_cfg=True, # U-Net: enables classifier free guidance
            embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=768, # U-Net: text mbedding features (default for T5-base)
            cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
        )

        self.lr = lr
    
    def forward(self, batch):
        audio_wave, text = batch
        return self.model(
            audio_wave,
            text=text, # Text conditioning, one element per batch
            embedding_mask_proba=0.1 # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train/loss', loss, batch_size=batch[0].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val/loss', loss, batch_size=batch[0].shape[0])

        # Log a sample to wandb
        columns = ["label", "generated audio"]
        data = []
        if batch_idx % 100 == 0:
            noise = torch.rand_like(batch[0])[0:1]
            label = batch[1][0]
            sample = self.model.sample(
                noise,
                text=[label],
                embedding_scale=15.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
                num_steps=10 # Higher for better quality, suggested num_steps: 10-100
            )
            data.append([
                label,
                wandb.Audio(sample[0, 0].cpu().numpy(), sample_rate=44100)
            ])
        wandb.log({'samples': wandb.Table(columns=columns, data=data)})

        return loss

# Strat run
run = wandb.init(reinit=True, project="fsd50k-diffusion")

# Init trainer
wandb_logger = WandbLogger(log_model='all')
checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min")
trainer = pl.Trainer(
    accelerator="gpu", devices=4, strategy="ddp", max_epochs=100,
    logger=wandb_logger, callbacks=[checkpoint_callback], limit_train_batches=10, limit_val_batches=5)

# Data loaders and model
model = FSD50KDiffusionModel()
train_dl = DataLoader(
    FSD50KCurated('../semaudio-few-shot/data/FSD50KSoundScapes/FSD50KScaperFmt/train'),
    batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(
    FSD50KCurated('../semaudio-few-shot/data/FSD50KSoundScapes/FSD50KScaperFmt/val'),
    batch_size=8, shuffle=False, collate_fn=collate_fn)

# Train
trainer.fit(model, train_dl, val_dl)

# End run
run.finish()
