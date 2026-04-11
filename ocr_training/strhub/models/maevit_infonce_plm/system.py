# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import math
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import Sequence, Any, Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from strhub.models.maevit_infonce_plm.modules import DecoderLayer, Decoder, TokenEmbedding
from strhub.models.maevit_infonce_plm.length_head import LengthHead, compute_length_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import numpy as np
import random

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from torch.optim.lr_scheduler import OneCycleLR

    
from einops import rearrange

from strhub.models.maevit_infonce_plm import clip


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class Model(CrossEntropySystem):

    GLYPH_CONFUSION_NEIGHBORS = {
        '0': ('1', '2', '6', '8', '9'),
        '1': ('0', '7', '8'),
        '2': ('0', '1', '8', '9'),
        '3': ('8',),
        '5': ('6', '8'),
        '6': ('0', '5', '8', '9'),
        '7': ('1',),
        '8': ('0', '1', '2', '3', '5', '6', '9'),
        '9': ('0', '2', '6', '8'),
        'G': ('I', 'O'),
        'H': ('I', 'N', 'Y'),
        'I': ('G', 'H', 'N', 'O', 'Y'),
        'N': ('H', 'I', 'O'),
        'O': ('G', 'I', 'N'),
        'Y': ('H', 'I'),
    }

    GLYPH_SINK_WEIGHTS = {
        '0': 0.55,
        '1': 0.35,
        '6': 0.65,
        '8': 0.80,
        '9': 0.50,
        'G': 0.30,
        'H': 0.25,
        'I': 0.45,
        'N': 0.25,
        'O': 0.30,
        'Y': 0.25,
    }

    DEFAULT_SUBSTITUTION_SOURCE_WEIGHTS = {
        ('6', '8'): 1.00,
        ('8', '6'): 0.90,
        ('9', '8'): 0.85,
        ('8', '9'): 0.70,
        ('1', '0'): 0.65,
        ('0', '1'): 0.60,
        ('5', '6'): 0.60,
        ('0', '8'): 0.60,
        ('3', '8'): 0.58,
        ('5', '8'): 0.54,
        ('0', '9'): 0.52,
        ('9', '0'): 0.48,
        ('1', '8'): 0.56,
        ('8', '1'): 0.52,
        ('6', '0'): 0.48,
        ('2', '0'): 0.42,
        ('2', '8'): 0.42,
        ('I', 'G'): 0.38,
        ('G', 'I'): 0.34,
        ('O', 'I'): 0.32,
        ('I', 'N'): 0.28,
        ('N', 'O'): 0.26,
        ('Y', 'H'): 0.24,
    }

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], embed_dim: int, mae_pretrained_path: str,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, 
                 use_selective_eos_aware_decoding: bool = False,
                 use_predicted_length_for_eos: bool = True,
                 eos_long_seq_threshold: int = 21,
                 eos_suppress_margin: int = 2,
                 eos_neutral_margin: int = 1,
                 eos_boost_margin: int = 2,
                 eos_suppress_bias: float = -2.0,
                 eos_boost_bias: float = 2.0,
                 use_uncertainty_conditioned_eos: bool = False,
                 uncertainty_type: str = 'mean_token_entropy',
                 uncertainty_threshold: float = 0.30,
                 eos_bias_scale_high_uncertainty: float = 1.0,
                 eos_bias_scale_low_uncertainty: float = 0.0,
                 use_length_bucket_gating: bool = False,
                 use_meta_gating: bool = False,
                 enable_eos_diagnostics: bool = False,
                 use_glyph_confusion_bias: bool = False,
                 glyph_confusion_margin_threshold: float = 1.25,
                 glyph_confusion_bias_scale: float = 0.75,
                 glyph_confusion_alt_boost: float = 0.35,
                 glyph_confusion_candidate_topk: int = 3,
                 use_substitution_confusion_bias: bool = False,
                 substitution_confusion_margin_threshold: float = 1.40,
                 substitution_confusion_bias_scale: float = 0.90,
                 substitution_confusion_alt_boost: float = 0.55,
                 substitution_confusion_candidate_topk: int = 4,
                 substitution_confusion_pairs_path: str = 'evaluation/results/M05/confusion_pairs_topk.csv',
                 use_substitution_train_aux: bool = False,
                 substitution_train_loss_weight: float = 0.15,
                 substitution_train_margin: float = 0.60,
                 substitution_train_topk: int = 12,
                 use_joint_glyph_train_aux: bool = False,
                 joint_glyph_loss_weight: float = 0.05,
                 joint_glyph_margin: float = 0.35,
                 use_length_head: bool = False, length_loss_weight: float = 0.1,
                 max_seq_length_for_head: int = 50, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        
        # Length head configuration
        self.use_length_head = use_length_head
        self.length_loss_weight = length_loss_weight
        self.max_seq_length_for_head = max_seq_length_for_head
        self.use_selective_eos_aware_decoding = use_selective_eos_aware_decoding
        self.use_predicted_length_for_eos = use_predicted_length_for_eos
        self.eos_long_seq_threshold = eos_long_seq_threshold
        self.eos_suppress_margin = eos_suppress_margin
        self.eos_neutral_margin = eos_neutral_margin
        self.eos_boost_margin = eos_boost_margin
        self.eos_suppress_bias = eos_suppress_bias
        self.eos_boost_bias = eos_boost_bias
        self.use_uncertainty_conditioned_eos = use_uncertainty_conditioned_eos
        self.uncertainty_type = uncertainty_type
        self.uncertainty_threshold = uncertainty_threshold
        self.eos_bias_scale_high_uncertainty = eos_bias_scale_high_uncertainty
        self.eos_bias_scale_low_uncertainty = eos_bias_scale_low_uncertainty
        self.use_length_bucket_gating = use_length_bucket_gating
        # Real-world inference usually lacks metadata; keep the flag for future expansion.
        self.use_meta_gating = use_meta_gating
        self.enable_eos_diagnostics = enable_eos_diagnostics
        self._eos_diag_summary = None
        self._last_eos_diag_batch = []
        self.use_glyph_confusion_bias = use_glyph_confusion_bias
        self.glyph_confusion_margin_threshold = glyph_confusion_margin_threshold
        self.glyph_confusion_bias_scale = glyph_confusion_bias_scale
        self.glyph_confusion_alt_boost = glyph_confusion_alt_boost
        self.glyph_confusion_candidate_topk = glyph_confusion_candidate_topk
        self.use_substitution_confusion_bias = use_substitution_confusion_bias
        self.substitution_confusion_margin_threshold = substitution_confusion_margin_threshold
        self.substitution_confusion_bias_scale = substitution_confusion_bias_scale
        self.substitution_confusion_alt_boost = substitution_confusion_alt_boost
        self.substitution_confusion_candidate_topk = substitution_confusion_candidate_topk
        self.substitution_confusion_pairs_path = substitution_confusion_pairs_path
        self.use_substitution_train_aux = use_substitution_train_aux
        self.substitution_train_loss_weight = substitution_train_loss_weight
        self.substitution_train_margin = substitution_train_margin
        self.substitution_train_topk = substitution_train_topk
        self.use_joint_glyph_train_aux = use_joint_glyph_train_aux
        self.joint_glyph_loss_weight = joint_glyph_loss_weight
        self.joint_glyph_margin = joint_glyph_margin

        self.coef_lr = kwargs["coef_lr"] if "coef_lr" in kwargs.keys() else 1.0
        self.coef_wd = kwargs["coef_wd"] if "coef_wd" in kwargs.keys() else 1.0
        self.image_freeze_nlayer = kwargs["image_freeze_nlayer"] if "image_freeze_nlayer" in kwargs.keys() else -1
        self.text_freeze_nlayer = kwargs["text_freeze_nlayer"] if "text_freeze_nlayer" in kwargs.keys() else -1

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        import strhub.models.models_mae as models_mae
        if img_size[0] == 32 and img_size[1] == 128:
            if embed_dim == 384:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_384_32x128')()
            elif embed_dim == 768:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_768_32x128')()
        elif img_size[0] == img_size[1] == 224:
            mae_model = getattr(models_mae, 'mae_vit_base_patch16_224x224')()

        chkpt_dir = mae_pretrained_path
        try:
            checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
        
        mae_model.load_state_dict(checkpoint['model'], strict=False)
        self.encoder = mae_model
        
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
            
        del self.clip_model.visual   
        if self.clip_model is not None:
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False
            self.clip_model.token_embedding.eval()  

            self.clip_model.positional_embedding.requires_grad = False 

            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False 
            self.clip_model.transformer.eval() 

            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = False 
            self.clip_model.ln_final.eval() 
            
        scale = embed_dim ** -0.5
        self.proj =  nn.Parameter(scale * torch.randn(embed_dim, 512))

        self.InfoNCELoss = InfoNCELoss()


        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        self._decode_token_strings = self.tokenizer._itos[:-2]
        self._decode_char_to_id = {
            token: idx for idx, token in enumerate(self._decode_token_strings)
            if token not in {self.tokenizer.EOS}
        }
        self._glyph_prior_bank = self._build_glyph_prior_bank()
        self._substitution_pairs = self._load_substitution_pairs()
        substitution_ids, substitution_weights, substitution_mask = self._build_substitution_aux_tables()
        glyph_ids, glyph_weights, glyph_mask = self._build_glyph_aux_tables()
        self.register_buffer('substitution_aux_ids', substitution_ids, persistent=False)
        self.register_buffer('substitution_aux_weights', substitution_weights, persistent=False)
        self.register_buffer('substitution_aux_mask', substitution_mask, persistent=False)
        self.register_buffer('glyph_aux_ids', glyph_ids, persistent=False)
        self.register_buffer('glyph_aux_weights', glyph_weights, persistent=False)
        self.register_buffer('glyph_aux_mask', glyph_mask, persistent=False)
        
        # Length prediction head (optional)
        if self.use_length_head:
            self.length_head = LengthHead(embed_dim, max_seq_length_for_head, dropout)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)
  

    def encode(self, img,labels = None):

        mae_memory = self.encoder(img)

        if labels is not None:
            labels = clip.tokenize(labels).to(mae_memory.device)
            text_features = self.clip_model.encode_text(labels) 
           
            loss = self.InfoNCELoss(mae_memory[:,0,:]@self.proj, text_features)

            return mae_memory,loss
        else:
            return mae_memory


    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape #B,tgt_len
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1)) #B*T*768
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1) #B*T*768
        tgt_query = self.dropout(tgt_query)#B*T*768
        return self.decoder(query=tgt_query, content=tgt_emb, memory=memory, query_mask=tgt_query_mask, content_mask=tgt_mask, content_key_padding_mask=tgt_padding_mask)

    def _use_selective_eos_bias(self, testing: bool) -> bool:
        return (
            testing
            and self.decode_ar
            and self.use_selective_eos_aware_decoding
            and self.use_predicted_length_for_eos
            and self.use_length_head
        )

    def reset_eos_diagnostics(self) -> None:
        self._eos_diag_summary = {
            'total_samples': 0,
            'num_samples_gating_enabled': 0,
            'num_samples_gating_disabled': 0,
            'num_samples_failed_uncertainty_gate': 0,
            'num_samples_failed_length_gate': 0,
            'num_samples_eos_bias_applied': 0,
            'num_samples_any_token_changed': 0,
            'num_samples_final_prediction_changed': 0,
            'total_decode_steps': 0,
            'num_steps_gating_active': 0,
            'num_steps_eos_logit_modified': 0,
            'num_steps_eos_rank_changed': 0,
            'num_steps_argmax_changed': 0,
        }
        self._last_eos_diag_batch = []

    def get_eos_diagnostics_summary(self) -> Dict[str, Any]:
        if self._eos_diag_summary is None:
            self.reset_eos_diagnostics()
        return dict(self._eos_diag_summary)

    def consume_last_eos_diagnostics_batch(self) -> List[Dict[str, Any]]:
        batch = self._last_eos_diag_batch
        self._last_eos_diag_batch = []
        return batch

    def _compute_uncertainty_score(
        self,
        step_logits: Tensor,
        running_entropy_sum: Optional[Tensor],
        running_max_prob_sum: Optional[Tensor],
        step_index: int,
    ) -> Tensor:
        probs = step_logits.softmax(-1)
        if self.uncertainty_type == 'mean_token_entropy':
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / math.log(probs.shape[-1])
            if running_entropy_sum is None:
                return entropy
            running_entropy_sum += entropy
            return running_entropy_sum / float(step_index + 1)
        if self.uncertainty_type == 'top1_top2_margin':
            top2 = probs.topk(k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            return 1.0 - margin
        if self.uncertainty_type == 'sequence_confidence':
            max_probs = probs.max(dim=-1).values
            if running_max_prob_sum is None:
                return 1.0 - max_probs
            running_max_prob_sum += max_probs
            return 1.0 - (running_max_prob_sum / float(step_index + 1))
        raise ValueError(f'Unsupported uncertainty_type: {self.uncertainty_type}')

    def _apply_selective_eos_bias(
        self,
        step_logits: Tensor,
        step_index: int,
        pred_lengths: Tensor,
        running_entropy_sum: Optional[Tensor],
        running_max_prob_sum: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        eos_logits_before = step_logits[:, self.eos_id].clone()
        argmax_before = step_logits.argmax(dim=-1)
        eos_rank_before = (step_logits > eos_logits_before.unsqueeze(1)).sum(dim=-1) + 1

        passed_length_gate = torch.ones(step_logits.shape[0], dtype=torch.bool, device=step_logits.device)
        if self.use_length_bucket_gating:
            passed_length_gate &= pred_lengths >= self.eos_long_seq_threshold
        active = passed_length_gate.clone()

        scale = torch.ones(step_logits.shape[0], device=step_logits.device)
        uncertainty = torch.zeros(step_logits.shape[0], device=step_logits.device)
        passed_uncertainty_gate = torch.ones(step_logits.shape[0], dtype=torch.bool, device=step_logits.device)
        if self.use_uncertainty_conditioned_eos:
            uncertainty = self._compute_uncertainty_score(
                step_logits, running_entropy_sum, running_max_prob_sum, step_index
            )
            high_uncertainty = uncertainty >= self.uncertainty_threshold
            passed_uncertainty_gate = high_uncertainty
            scale = torch.where(
                high_uncertainty,
                torch.full_like(scale, self.eos_bias_scale_high_uncertainty),
                torch.full_like(scale, self.eos_bias_scale_low_uncertainty),
            )
            active &= passed_uncertainty_gate & (scale != 0)

        bias = torch.zeros(step_logits.shape[0], device=step_logits.device)
        suppress_mask = step_index < (pred_lengths - self.eos_suppress_margin)
        neutral_mask = (pred_lengths - step_index).abs() <= self.eos_neutral_margin
        boost_mask = step_index > (pred_lengths + self.eos_boost_margin)
        bias = torch.where(
            suppress_mask,
            torch.full_like(bias, self.eos_suppress_bias),
            bias,
        )
        bias = torch.where(
            boost_mask & ~neutral_mask,
            torch.full_like(bias, self.eos_boost_bias),
            bias,
        )
        bias = bias * scale
        applied_bias = torch.where(active, bias, torch.zeros_like(bias))
        step_logits[:, self.eos_id] = step_logits[:, self.eos_id] + applied_bias

        eos_logits_after = step_logits[:, self.eos_id]
        argmax_after = step_logits.argmax(dim=-1)
        eos_rank_after = (step_logits > eos_logits_after.unsqueeze(1)).sum(dim=-1) + 1
        diag = {
            'uncertainty': uncertainty,
            'passed_length_gate': passed_length_gate,
            'passed_uncertainty_gate': passed_uncertainty_gate,
            'gating_active': active,
            'eos_logit_modified': applied_bias != 0,
            'eos_rank_changed': eos_rank_before != eos_rank_after,
            'argmax_changed': argmax_before != argmax_after,
        }
        return step_logits, diag

    def _apply_glyph_confusion_bias(self, step_logits: Tensor) -> Tensor:
        if not self.use_glyph_confusion_bias:
            return step_logits

        topk = min(self.glyph_confusion_candidate_topk, step_logits.shape[-1])
        if topk < 2:
            return step_logits

        top_vals, top_ids = step_logits.topk(k=topk, dim=-1)
        for batch_idx in range(step_logits.shape[0]):
            top1_id = int(top_ids[batch_idx, 0].item())
            top1_char = self._decode_token_strings[top1_id]
            if top1_char == self.tokenizer.EOS:
                continue

            neighbors = self.GLYPH_CONFUSION_NEIGHBORS.get(top1_char)
            sink_weight = self.GLYPH_SINK_WEIGHTS.get(top1_char, 0.0)
            if not neighbors or sink_weight <= 0:
                continue

            for alt_rank in range(1, topk):
                alt_id = int(top_ids[batch_idx, alt_rank].item())
                alt_char = self._decode_token_strings[alt_id]
                if alt_char not in neighbors:
                    continue

                margin = float(top_vals[batch_idx, 0].item() - top_vals[batch_idx, alt_rank].item())
                if margin >= self.glyph_confusion_margin_threshold:
                    continue

                prototype_weight = float(self._glyph_prior_bank[top1_id, alt_id].item())
                if prototype_weight <= 0:
                    continue
                scaled = (1.0 - (margin / max(self.glyph_confusion_margin_threshold, 1e-6)))
                penalty = sink_weight * prototype_weight * self.glyph_confusion_bias_scale * scaled
                if penalty <= 0:
                    continue

                step_logits[batch_idx, top1_id] -= penalty
                step_logits[batch_idx, alt_id] += penalty * self.glyph_confusion_alt_boost
                break

        return step_logits

    def _build_glyph_prior_bank(self) -> Tensor:
        size = len(self._decode_token_strings)
        bank = torch.zeros(size, size, dtype=torch.float32)
        for sink_char, neighbors in self.GLYPH_CONFUSION_NEIGHBORS.items():
            sink_id = self._decode_char_to_id.get(sink_char)
            if sink_id is None:
                continue
            sink_weight = self.GLYPH_SINK_WEIGHTS.get(sink_char, 0.0)
            for src_char in neighbors:
                src_id = self._decode_char_to_id.get(src_char)
                if src_id is None:
                    continue
                reverse = 1.0 if sink_char in self.GLYPH_CONFUSION_NEIGHBORS.get(src_char, ()) else 0.0
                bank[sink_id, src_id] = max(bank[sink_id, src_id], sink_weight * (0.7 + 0.3 * reverse))
        return bank

    def _resolve_substitution_pairs_path(self) -> Optional[Path]:
        raw_path = Path(self.substitution_confusion_pairs_path)
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            project_root = Path(__file__).resolve().parents[3]
            candidates.extend([
                Path.cwd() / raw_path,
                project_root / raw_path,
            ])
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_substitution_pairs(self) -> Dict[Tuple[str, str], float]:
        pairs = dict(self.DEFAULT_SUBSTITUTION_SOURCE_WEIGHTS)
        path = self._resolve_substitution_pairs_path()
        if path is None:
            return pairs

        directional_counts: Dict[Tuple[str, str], int] = {}
        try:
            with path.open('r', encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('subset_name') != 'overall':
                        continue
                    src_char = row.get('src_char', '')
                    tgt_char = row.get('tgt_char', '')
                    if not src_char or not tgt_char:
                        continue
                    count = int(row.get('count', 0))
                    if count <= 0:
                        continue
                    directional_counts[(src_char, tgt_char)] = count
        except Exception:
            return pairs

        if not directional_counts:
            return pairs

        max_count = max(directional_counts.values())
        for pair, count in directional_counts.items():
            src_char, tgt_char = pair
            if src_char not in self._decode_char_to_id or tgt_char not in self._decode_char_to_id:
                continue
            normalized = count / max_count
            pairs[pair] = max(pairs.get(pair, 0.0), normalized)
        return pairs

    def _apply_substitution_confusion_bias(self, step_logits: Tensor) -> Tensor:
        if not self.use_substitution_confusion_bias:
            return step_logits

        topk = min(self.substitution_confusion_candidate_topk, step_logits.shape[-1])
        if topk < 2:
            return step_logits

        top_vals, top_ids = step_logits.topk(k=topk, dim=-1)
        for batch_idx in range(step_logits.shape[0]):
            sink_id = int(top_ids[batch_idx, 0].item())
            sink_char = self._decode_token_strings[sink_id]
            if sink_char == self.tokenizer.EOS:
                continue

            best_candidate = None
            for alt_rank in range(1, topk):
                src_id = int(top_ids[batch_idx, alt_rank].item())
                src_char = self._decode_token_strings[src_id]
                pair_weight = self._substitution_pairs.get((src_char, sink_char), 0.0)
                if pair_weight <= 0:
                    continue
                margin = float(top_vals[batch_idx, 0].item() - top_vals[batch_idx, alt_rank].item())
                if margin >= self.substitution_confusion_margin_threshold:
                    continue
                score = pair_weight - 0.1 * margin
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, src_id, margin, pair_weight)

            if best_candidate is None:
                continue

            _, src_id, margin, pair_weight = best_candidate
            scaled = (1.0 - (margin / max(self.substitution_confusion_margin_threshold, 1e-6)))
            penalty = pair_weight * self.substitution_confusion_bias_scale * scaled
            if penalty <= 0:
                continue
            step_logits[batch_idx, sink_id] -= penalty
            step_logits[batch_idx, src_id] += penalty * self.substitution_confusion_alt_boost

        return step_logits

    def _build_substitution_aux_tables(self) -> Tuple[Tensor, Tensor, Tensor]:
        size = len(self._decode_token_strings)
        grouped: Dict[int, List[Tuple[int, float]]] = {}
        for (src_char, tgt_char), weight in self._substitution_pairs.items():
            src_id = self._decode_char_to_id.get(src_char)
            tgt_id = self._decode_char_to_id.get(tgt_char)
            if src_id is None or tgt_id is None or src_id == tgt_id or weight <= 0:
                continue
            grouped.setdefault(src_id, []).append((tgt_id, float(weight)))

        max_neighbors = max((min(len(v), self.substitution_train_topk) for v in grouped.values()), default=1)
        ids = torch.zeros(size, max_neighbors, dtype=torch.long)
        weights = torch.zeros(size, max_neighbors, dtype=torch.float32)
        mask = torch.zeros(size, max_neighbors, dtype=torch.bool)
        for src_id, items in grouped.items():
            items = sorted(items, key=lambda x: x[1], reverse=True)[:max_neighbors]
            total = sum(weight for _, weight in items)
            for idx, (tgt_id, weight) in enumerate(items):
                ids[src_id, idx] = tgt_id
                weights[src_id, idx] = weight / total if total > 0 else 0.0
                mask[src_id, idx] = True
        return ids, weights, mask

    def _build_glyph_aux_tables(self) -> Tuple[Tensor, Tensor, Tensor]:
        size = len(self._decode_token_strings)
        grouped: Dict[int, List[Tuple[int, float]]] = {}
        for src_char, src_id in self._decode_char_to_id.items():
            neighbors = set(self.GLYPH_CONFUSION_NEIGHBORS.get(src_char, ()))
            for candidate, candidate_neighbors in self.GLYPH_CONFUSION_NEIGHBORS.items():
                if src_char in candidate_neighbors:
                    neighbors.add(candidate)
            for tgt_char in neighbors:
                tgt_id = self._decode_char_to_id.get(tgt_char)
                if tgt_id is None or tgt_id == src_id:
                    continue
                weight = float(max(self._glyph_prior_bank[src_id, tgt_id].item(), self._glyph_prior_bank[tgt_id, src_id].item()))
                if weight <= 0:
                    weight = 0.25
                grouped.setdefault(src_id, []).append((tgt_id, weight))

        max_neighbors = max((len(v) for v in grouped.values()), default=1)
        ids = torch.zeros(size, max_neighbors, dtype=torch.long)
        weights = torch.zeros(size, max_neighbors, dtype=torch.float32)
        mask = torch.zeros(size, max_neighbors, dtype=torch.bool)
        for src_id, items in grouped.items():
            dedup: Dict[int, float] = {}
            for tgt_id, weight in items:
                dedup[tgt_id] = max(dedup.get(tgt_id, 0.0), weight)
            ordered = sorted(dedup.items(), key=lambda x: x[1], reverse=True)
            total = sum(weight for _, weight in ordered)
            for idx, (tgt_id, weight) in enumerate(ordered):
                ids[src_id, idx] = tgt_id
                weights[src_id, idx] = weight / total if total > 0 else 0.0
                mask[src_id, idx] = True
        return ids, weights, mask

    def _compute_substitution_aux_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if not self.use_substitution_train_aux:
            return logits.new_zeros(())

        flat_logits = logits.flatten(end_dim=1)
        flat_targets = targets.flatten()
        valid = (flat_targets != self.pad_id) & (flat_targets < self.substitution_aux_ids.shape[0])
        if not valid.any():
            return flat_logits.new_zeros(())

        flat_logits = flat_logits[valid]
        flat_targets = flat_targets[valid]
        gt_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        conf_ids = self.substitution_aux_ids[flat_targets]
        conf_weights = self.substitution_aux_weights[flat_targets]
        conf_mask = self.substitution_aux_mask[flat_targets]
        conf_logits = flat_logits.gather(1, conf_ids)
        margin_gap = self.substitution_train_margin - (gt_logits.unsqueeze(1) - conf_logits)
        penalties = F.relu(margin_gap) * conf_weights * conf_mask.float()
        denom = conf_mask.float().sum().clamp_min(1.0)
        return penalties.sum() / denom

    def _compute_joint_glyph_aux_loss(self, hidden: Tensor, targets: Tensor) -> Tensor:
        if not self.use_joint_glyph_train_aux:
            return hidden.new_zeros(())

        flat_hidden = hidden.flatten(end_dim=1)
        flat_targets = targets.flatten()
        valid = (flat_targets != self.pad_id) & (flat_targets < self.glyph_aux_ids.shape[0])
        if not valid.any():
            return flat_hidden.new_zeros(())

        flat_hidden = F.normalize(flat_hidden[valid], dim=-1)
        flat_targets = flat_targets[valid]
        proto = F.normalize(self.head.weight, dim=-1)
        gt_proto = proto[flat_targets]
        gt_sim = (flat_hidden * gt_proto).sum(dim=-1)
        conf_ids = self.glyph_aux_ids[flat_targets]
        conf_weights = self.glyph_aux_weights[flat_targets]
        conf_mask = self.glyph_aux_mask[flat_targets]
        conf_proto = proto[conf_ids]
        conf_sim = (flat_hidden.unsqueeze(1) * conf_proto).sum(dim=-1)
        margin_gap = self.joint_glyph_margin - (gt_sim.unsqueeze(1) - conf_sim)
        penalties = F.relu(margin_gap) * conf_weights * conf_mask.float()
        denom = conf_mask.float().sum().clamp_min(1.0)
        return penalties.sum() / denom
   
        
    def forward(self, images, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length) #25
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1 
        
        memory = self.encode(images) 
        use_eos_bias = self._use_selective_eos_bias(testing)
        diagnostics_active = testing and use_eos_bias and self.enable_eos_diagnostics
        pred_lengths = None
        running_entropy_sum = None
        running_max_prob_sum = None
        if use_eos_bias:
            pred_lengths = self.length_head.predict(memory).clamp(max=self.max_label_length).to(self._device)
            if self.uncertainty_type == 'mean_token_entropy':
                running_entropy_sum = torch.zeros(bs, device=self._device)
            elif self.uncertainty_type == 'sequence_confidence':
                running_max_prob_sum = torch.zeros(bs, device=self._device)
        if diagnostics_active and self._eos_diag_summary is None:
            self.reset_eos_diagnostics()
        if diagnostics_active:
            sample_uncertainty_sum = torch.zeros(bs, device=self._device)
            sample_uncertainty_count = torch.zeros(bs, device=self._device)
            sample_passed_length_gate = torch.ones(bs, dtype=torch.bool, device=self._device)
            if self.use_length_bucket_gating:
                sample_passed_length_gate &= pred_lengths >= self.eos_long_seq_threshold
            sample_passed_uncertainty_gate = torch.zeros(bs, dtype=torch.bool, device=self._device)
            sample_gating_enabled = torch.zeros(bs, dtype=torch.bool, device=self._device)
            sample_eos_bias_applied = torch.zeros(bs, dtype=torch.bool, device=self._device)
            sample_any_token_changed = torch.zeros(bs, dtype=torch.bool, device=self._device)
            sample_steps_modified = torch.zeros(bs, dtype=torch.long, device=self._device)
            sample_steps_argmax_changed = torch.zeros(bs, dtype=torch.long, device=self._device)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.tokenizer.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                # the next token probability is in the output's ith token position
                step_logits = self.head(tgt_out).squeeze(1)
                if use_eos_bias:
                    step_logits, diag = self._apply_selective_eos_bias(
                        step_logits, i, pred_lengths, running_entropy_sum, running_max_prob_sum
                    )
                    if diagnostics_active:
                        self._eos_diag_summary['total_decode_steps'] += bs
                        self._eos_diag_summary['num_steps_gating_active'] += int(diag['gating_active'].sum().item())
                        self._eos_diag_summary['num_steps_eos_logit_modified'] += int(diag['eos_logit_modified'].sum().item())
                        self._eos_diag_summary['num_steps_eos_rank_changed'] += int(diag['eos_rank_changed'].sum().item())
                        self._eos_diag_summary['num_steps_argmax_changed'] += int(diag['argmax_changed'].sum().item())
                        sample_passed_uncertainty_gate |= diag['passed_uncertainty_gate']
                        sample_gating_enabled |= diag['gating_active']
                        sample_eos_bias_applied |= diag['eos_logit_modified']
                        sample_any_token_changed |= diag['argmax_changed']
                        sample_steps_modified += diag['eos_logit_modified'].long()
                        sample_steps_argmax_changed += diag['argmax_changed'].long()
                        if self.use_uncertainty_conditioned_eos:
                            sample_uncertainty_sum += diag['uncertainty']
                            sample_uncertainty_count += 1
                step_logits = self._apply_glyph_confusion_bias(step_logits)
                step_logits = self._apply_substitution_confusion_bias(step_logits)
                p_i = step_logits.unsqueeze(1)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = step_logits.argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.tokenizer.eos_id).any(dim=-1).all():
                        
                        break
            t = len(logits)                
            logits = torch.cat(logits, dim=1)
            if diagnostics_active:
                if not self.use_uncertainty_conditioned_eos:
                    sample_passed_uncertainty_gate[:] = True
                mean_uncertainty = torch.where(
                    sample_uncertainty_count > 0,
                    sample_uncertainty_sum / sample_uncertainty_count.clamp_min(1),
                    torch.zeros_like(sample_uncertainty_sum),
                )
                batch_diag = []
                for b in range(bs):
                    batch_diag.append({
                        'pred_len_from_head': int(pred_lengths[b].item()),
                        'uncertainty_value': float(mean_uncertainty[b].item()),
                        'gating_enabled': bool(sample_gating_enabled[b].item()),
                        'passed_uncertainty_gate': bool(sample_passed_uncertainty_gate[b].item()),
                        'passed_length_gate': bool(sample_passed_length_gate[b].item()),
                        'eos_bias_applied': bool(sample_eos_bias_applied[b].item()),
                        'num_steps_modified': int(sample_steps_modified[b].item()),
                        'num_steps_argmax_changed': int(sample_steps_argmax_changed[b].item()),
                        'any_token_changed': bool(sample_any_token_changed[b].item()),
                    })
                self._last_eos_diag_batch = batch_diag
                self._eos_diag_summary['total_samples'] += bs
                self._eos_diag_summary['num_samples_gating_enabled'] += sum(x['gating_enabled'] for x in batch_diag)
                self._eos_diag_summary['num_samples_gating_disabled'] += bs - sum(x['gating_enabled'] for x in batch_diag)
                self._eos_diag_summary['num_samples_failed_uncertainty_gate'] += sum(not x['passed_uncertainty_gate'] for x in batch_diag)
                self._eos_diag_summary['num_samples_failed_length_gate'] += sum(not x['passed_length_gate'] for x in batch_diag)
                self._eos_diag_summary['num_samples_eos_bias_applied'] += sum(x['eos_bias_applied'] for x in batch_diag)
                self._eos_diag_summary['num_samples_any_token_changed'] += sum(x['any_token_changed'] for x in batch_diag)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self._device)
            num_steps = t
            tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == self.tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.head(tgt_out)

        return logits
        

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device) # B * (T+1)

        memory,cross_modal_loss = self.encode(images,labels) # B*197*768
        
        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt) # k*(T+1)
        tgt_in = tgt[:, :-1] #B*32
        tgt_out = tgt[:, 1:] #B*32
        # # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)#B*32

        ocr_loss = 0
        substitution_aux_loss = 0
        joint_glyph_aux_loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm) 
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask) # B * T * 768
            step_logits = self.head(out)
            logits = step_logits.flatten(end_dim=1)  #(B * len(label+1),NUM_CLASS)
            ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            substitution_aux_loss += n * self._compute_substitution_aux_loss(step_logits, tgt_out)
            joint_glyph_aux_loss += n * self._compute_joint_glyph_aux_loss(out, tgt_out)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        
        step_logits = self.head(out)
        logits = step_logits.flatten(end_dim=1)
        ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        substitution_aux_loss += n * self._compute_substitution_aux_loss(step_logits, tgt_out)
        joint_glyph_aux_loss += n * self._compute_joint_glyph_aux_loss(out, tgt_out)
        loss_numel += n        
        ocr_loss /= loss_numel
        substitution_aux_loss /= loss_numel
        joint_glyph_aux_loss /= loss_numel
        
        # Length prediction loss (optional)
        length_loss = 0.0
        if self.use_length_head:
            # Compute ground truth lengths (excluding <bos> and <eos>)
            gt_lengths = (tgt_out != self.pad_id).sum(dim=1).clamp(max=self.max_seq_length_for_head)
            # Predict lengths from encoder features
            length_logits = self.length_head(memory)
            length_loss = compute_length_loss(length_logits, gt_lengths)
            # Log length prediction accuracy
            pred_lengths = length_logits.argmax(dim=-1)
            length_acc = (pred_lengths == gt_lengths).float().mean()
            self.log('length_loss', length_loss)
            self.log('length_acc', length_acc * 100)
        
        # Total loss
        loss = ocr_loss + 0.1 * cross_modal_loss
        if self.use_length_head:
            loss = loss + self.length_loss_weight * length_loss
        if self.use_substitution_train_aux:
            loss = loss + self.substitution_train_loss_weight * substitution_aux_loss
        if self.use_joint_glyph_train_aux:
            loss = loss + self.joint_glyph_loss_weight * joint_glyph_aux_loss

        self.log('loss', loss)
        self.log('ocr_loss', ocr_loss)
        self.log('cross_modal_loss', cross_modal_loss)
        if self.use_substitution_train_aux:
            self.log('substitution_aux_loss', substitution_aux_loss)
        if self.use_joint_glyph_train_aux:
            self.log('joint_glyph_aux_loss', joint_glyph_aux_loss)

        return loss
    
    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * (self.trainer.num_devices * self.batch_size) / 512  
        lr = lr_scale * self.lr

        # https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/training/main.py#L171
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n \
                            or "pos_queries" in n or "text_embed" in n
        include = lambda n, p: not exclude(n, p)

        encoder_params = list(self.encoder.named_parameters())
        enc_gain_or_bias_params = [p for n, p in encoder_params if exclude(n, p) and p.requires_grad]
        enc_rest_params = [p for n, p in encoder_params if include(n, p) and p.requires_grad]


        # decoder parameters
        decoder_params = [(n, p) for n, p in list(self.named_parameters()) if "encoder" not in n and "clip_model" not in n] 
        dec_gain_or_bias_params = [p for n, p in decoder_params if exclude(n, p) and p.requires_grad]
        dec_rest_params = [p for n, p in decoder_params if include(n, p) and p.requires_grad]


        optimizer = torch.optim.AdamW(
            [
                {"params": enc_gain_or_bias_params, "weight_decay": 0., 'lr': lr},
                {"params": enc_rest_params, "weight_decay": self.weight_decay, 'lr': lr},
                {"params": dec_gain_or_bias_params, "weight_decay": 0., 'lr': lr * self.coef_lr},
                {"params": dec_rest_params, "weight_decay": self.weight_decay * self.coef_wd, 'lr': lr * self.coef_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1.0e-6,
            )
        sched = OneCycleLR(optimizer, [lr,lr,lr * self.coef_lr, lr * self.coef_lr],
                            self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                            cycle_momentum=False)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
