from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from strhub.models.base import CTCSystem


class Model(CTCSystem):

    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        embed_dim: int,
        mae_pretrained_path: str,
        dropout: float = 0.0,
        adapter_mode: str = 'none',
        adapter_dim: int = 512,
        sequence_neck: str = 'none',
        sequence_neck_hidden: int = 256,
        blank_bias_init: float = 0.0,
        logit_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length

        import strhub.models.models_mae as models_mae

        if img_size[0] == 32 and img_size[1] == 128:
            if embed_dim == 384:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_384_32x128')()
            elif embed_dim == 768:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_768_32x128')()
            else:
                raise ValueError(f'Unsupported embed_dim for 32x128 MAE CTC probe: {embed_dim}')
        elif img_size[0] == img_size[1] == 224:
            mae_model = getattr(models_mae, 'mae_vit_base_patch16_224x224')()
        else:
            raise ValueError(f'Unsupported img_size for MAE CTC probe: {img_size}')

        try:
            checkpoint = torch.load(mae_pretrained_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(mae_pretrained_path, map_location='cpu')

        mae_model.load_state_dict(checkpoint['model'], strict=False)
        self.encoder = mae_model
        self.dropout = nn.Dropout(p=dropout)
        self.adapter = self._build_adapter(adapter_mode, embed_dim, adapter_dim)
        self.sequence_neck = self._build_sequence_neck(sequence_neck, embed_dim, sequence_neck_hidden)
        head_dim = embed_dim if sequence_neck != 'bilstm' else sequence_neck_hidden * 2
        self.head = nn.Linear(head_dim, len(self.tokenizer))
        self.logit_temperature = float(logit_temperature)
        if self.logit_temperature <= 0:
            raise ValueError(f'logit_temperature must be > 0, got {self.logit_temperature}')
        if self.head.bias is not None and abs(blank_bias_init) > 0:
            with torch.no_grad():
                self.head.bias[self.blank_id] = float(blank_bias_init)

    @staticmethod
    def _build_adapter(adapter_mode: str, embed_dim: int, adapter_dim: int) -> nn.Module:
        if adapter_mode == 'none':
            return nn.Identity()
        if adapter_mode == 'ln_linear':
            return nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
        if adapter_mode == 'proj_ln_linear':
            return nn.Sequential(
                nn.Linear(embed_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.Linear(adapter_dim, embed_dim),
            )
        raise ValueError(f'Unsupported adapter_mode for MAE CTC probe: {adapter_mode}')

    @staticmethod
    def _build_sequence_neck(sequence_neck: str, embed_dim: int, sequence_neck_hidden: int) -> nn.Module:
        if sequence_neck == 'none':
            return nn.Identity()
        if sequence_neck == 'bilstm':
            return nn.LSTM(
                input_size=embed_dim,
                hidden_size=sequence_neck_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        raise ValueError(f'Unsupported sequence_neck for MAE CTC probe: {sequence_neck}')

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def _row_major_sequenceize(self, memory: Tensor) -> Tensor:
        patch_tokens = memory[:, 1:, :]
        grid_h, grid_w = self.encoder.patch_embed.grid_size
        batch_size, num_tokens, dim = patch_tokens.shape
        expected_tokens = grid_h * grid_w
        if num_tokens != expected_tokens:
            raise ValueError(
                f'Unexpected patch token count for row-major CTC probe: got {num_tokens}, '
                f'expected {expected_tokens} ({grid_h}x{grid_w}).'
            )
        return patch_tokens.reshape(batch_size, grid_h, grid_w, dim).reshape(batch_size, expected_tokens, dim)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        del max_length
        memory = self.encode(images)
        seq = self._row_major_sequenceize(memory)
        seq = self.adapter(seq)
        seq = self.dropout(seq)
        if isinstance(self.sequence_neck, nn.LSTM):
            seq, _ = self.sequence_neck(seq)
        else:
            seq = self.sequence_neck(seq)
        logits = self.head(seq)
        if self.logit_temperature != 1.0:
            logits = logits / self.logit_temperature
        return logits

    def training_step(self, batch, batch_idx):
        del batch_idx
        images, labels = batch
        _, loss, loss_numel = self.forward_logits_loss(images, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
        self.log('train_loss_numel', float(loss_numel), on_step=True, on_epoch=False, batch_size=images.shape[0])
        return loss
