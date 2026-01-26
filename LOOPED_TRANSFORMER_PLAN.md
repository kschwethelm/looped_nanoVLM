# Looped Transformer Implementation Plan for nanoVLM

## Overview

A **looped transformer** (also known as "Universal Transformer" or "weight-tied transformer") is an architecture where a single transformer block (or small set of blocks) is reused/iterated multiple times, rather than having N separate blocks with unique parameters. This dramatically reduces parameter count while potentially maintaining or improving performance through iterative refinement.

### Current Architecture Summary

| Component | Blocks | Parameters (approx) |
|-----------|--------|---------------------|
| Vision Encoder (ViT) | 12 unique blocks | ~85M |
| Language Model | 32 unique blocks | ~360M |
| Modality Projector | 1 linear layer | ~12M |

---

## Phase 1: Configuration System Updates

### Files to Modify
- `models/config.py`

### Changes Required

Add new configuration parameters to `VLMConfig`:

```python
# Looped Transformer Configuration
lm_looped: bool = False                    # Enable looped transformer for LM
lm_n_core_blocks: int = 4                  # Number of unique blocks (core)
lm_n_loops: int = 8                        # Number of loop iterations
lm_loop_embedding_dim: int = 64            # Dimension for loop position embedding
lm_progressive_loops: bool = False         # Enable progressive loop training
lm_adaptive_exit: bool = False             # Enable early exit mechanism

vit_looped: bool = False                   # Enable looped transformer for ViT
vit_n_core_blocks: int = 3                 # Number of unique ViT blocks
vit_n_loops: int = 4                       # Number of ViT loop iterations
```

### Design Rationale
- Keeping `lm_n_blocks` allows backward compatibility (32 blocks = 4 core × 8 loops)
- Separate flags for ViT and LM allow independent experimentation
- Progressive loops supports curriculum learning

---

## Phase 2: Language Model Looped Architecture

### Files to Modify
- `models/language_model.py`

### 2.1 Create Loop Position Embedding

Add a new class to inject iteration/loop information:

```python
class LoopPositionEmbedding(nn.Module):
    """Encodes which loop iteration the model is currently in."""
    def __init__(self, cfg):
        super().__init__()
        self.n_loops = cfg.lm_n_loops
        self.hidden_dim = cfg.lm_hidden_dim
        # Learnable embedding per loop iteration
        self.loop_emb = nn.Embedding(self.n_loops, self.hidden_dim)

    def forward(self, loop_idx: int) -> torch.Tensor:
        """Returns embedding for the given loop iteration."""
        idx = torch.tensor([loop_idx], device=self.loop_emb.weight.device)
        return self.loop_emb(idx)  # Shape: [1, hidden_dim]
```

**Design Choices:**
1. **Additive**: Add loop embedding to hidden states (simplest)
2. **FiLM**: Scale and shift hidden states based on loop (more expressive)
3. **Concatenate**: Expand hidden dim (breaks compatibility)

**Recommendation**: Start with additive, then experiment with FiLM.

### 2.2 Modify LanguageModel Class

Transform the block iteration from:
```python
# Current (lines 399-401, 471-472)
self.blocks = nn.ModuleList([
    LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
])
for i, block in enumerate(self.blocks):
    x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
```

To a looped structure:
```python
# Proposed
def __init__(self, cfg):
    ...
    if cfg.lm_looped:
        self.core_blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_core_blocks)
        ])
        self.loop_embedding = LoopPositionEmbedding(cfg)
        self.n_loops = cfg.lm_n_loops
        self.looped = True
    else:
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.looped = False
    ...

def forward(self, x, attention_mask=None, kv_cache=None, start_pos=0):
    ...
    if self.looped:
        # Looped forward pass
        total_virtual_layers = self.n_loops * len(self.core_blocks)
        if kv_cache is None:
            kv_cache = [None] * total_virtual_layers

        for loop_idx in range(self.n_loops):
            loop_emb = self.loop_embedding(loop_idx)
            for block_idx, block in enumerate(self.core_blocks):
                cache_idx = loop_idx * len(self.core_blocks) + block_idx
                x = x + loop_emb  # Additive loop embedding
                x, kv_cache[cache_idx] = block(x, cos, sin, attention_mask, kv_cache[cache_idx])
    else:
        # Original forward pass
        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
    ...
```

### 2.3 KV Cache Adaptation

**Challenge**: In standard transformers, each block has its own KV cache. In looped transformers, the same block processes different "virtual layers."

**Options:**
1. **Separate cache per virtual layer**: `n_loops × n_core_blocks` caches (current approach scaled)
2. **Shared cache across loops**: Single cache that gets updated each loop (recurrent style)
3. **Hybrid**: Some layers share, some don't

**Recommendation**: Start with Option 1 (separate cache per virtual layer) for simplicity and to match the effective depth.

### 2.4 Residual Connections

Consider whether to use:
- **Dense residuals**: Each loop gets a skip connection from input
- **Standard residuals**: Sequential as current (recommended to start)
- **Highway connections**: Gated residuals

---

## Phase 3: Vision Transformer Looped Architecture

### Files to Modify
- `models/vision_transformer.py`

### 3.1 Modify ViT Class

Similar changes to `ViT` class (lines 131-168):

```python
class ViTLoopPositionEmbedding(nn.Module):
    """Encodes which loop iteration the ViT is currently in."""
    def __init__(self, cfg):
        super().__init__()
        self.n_loops = cfg.vit_n_loops
        self.hidden_dim = cfg.vit_hidden_dim
        self.loop_emb = nn.Embedding(self.n_loops, self.hidden_dim)

    def forward(self, loop_idx: int) -> torch.Tensor:
        idx = torch.tensor([loop_idx], device=self.loop_emb.weight.device)
        return self.loop_emb(idx)

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...
        if cfg.vit_looped:
            self.core_blocks = nn.ModuleList([
                ViTBlock(cfg) for _ in range(cfg.vit_n_core_blocks)
            ])
            self.loop_embedding = ViTLoopPositionEmbedding(cfg)
            self.n_loops = cfg.vit_n_loops
            self.looped = True
        else:
            self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
            self.looped = False
        ...

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)

        if self.looped:
            for loop_idx in range(self.n_loops):
                loop_emb = self.loop_embedding(loop_idx)
                for block in self.core_blocks:
                    x = x + loop_emb
                    x = block(x)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.layer_norm(x)
        return x
```

**Note**: ViT uses bidirectional attention, no KV cache needed during inference, making the looped implementation simpler.

---

## Phase 4: Pretrained Weight Loading Strategy

### Files to Modify
- `models/language_model.py` (`from_pretrained` method, lines 538-682)
- `models/vision_transformer.py` (`from_pretrained` method, lines 171-251)

### Strategies for Weight Initialization

#### Option A: Average Weights
Average weights from groups of pretrained layers into core blocks:
```python
# For 4 core blocks from 32 pretrained blocks
# Each core block gets averaged weights from 8 layers
for core_idx in range(n_core_blocks):
    layers_to_average = [i for i in range(core_idx, 32, n_core_blocks)]
    # E.g., core_block[0] averages layers [0, 4, 8, 12, 16, 20, 24, 28]
    averaged_weights = {}
    for key in pretrained_blocks[0].state_dict().keys():
        tensors = [pretrained_blocks[i].state_dict()[key] for i in layers_to_average]
        averaged_weights[key] = torch.stack(tensors).mean(dim=0)
    core_blocks[core_idx].load_state_dict(averaged_weights)
```

#### Option B: First N Layers
Use the first N pretrained layers as core blocks (simpler):
```python
for core_idx in range(n_core_blocks):
    core_blocks[core_idx].load_state_dict(pretrained_blocks[core_idx].state_dict())
```

#### Option C: Evenly Spaced Layers
Select evenly spaced layers:
```python
# For 4 core blocks from 32: use layers [0, 8, 16, 24]
indices = [i * (32 // n_core_blocks) for i in range(n_core_blocks)]
for core_idx, pretrained_idx in enumerate(indices):
    core_blocks[core_idx].load_state_dict(pretrained_blocks[pretrained_idx].state_dict())
```

#### Option D: Train from Scratch
Initialize randomly and train the looped model from scratch.

**Recommendation**: Start with Option B or C for faster convergence, then experiment with Option D for potentially better final performance.

### Implementation in `from_pretrained`:

```python
@classmethod
def from_pretrained(cls, cfg):
    ...
    model = cls(cfg)

    if cfg.lm_looped:
        # Load weights for looped architecture
        # Choose one of the strategies above
        pretrained_blocks_sd = load_pretrained_blocks(cfg)

        if cfg.lm_looped_init_strategy == 'first_n':
            for i in range(cfg.lm_n_core_blocks):
                copy_block_weights(pretrained_blocks_sd[i], model.core_blocks[i])
        elif cfg.lm_looped_init_strategy == 'evenly_spaced':
            indices = [i * (len(pretrained_blocks_sd) // cfg.lm_n_core_blocks)
                      for i in range(cfg.lm_n_core_blocks)]
            for core_idx, pretrained_idx in enumerate(indices):
                copy_block_weights(pretrained_blocks_sd[pretrained_idx], model.core_blocks[core_idx])
        elif cfg.lm_looped_init_strategy == 'average':
            # Average groups of layers
            ...

        # Initialize loop embeddings from scratch
        nn.init.normal_(model.loop_embedding.loop_emb.weight, std=0.02)
    else:
        # Original loading logic
        ...
```

---

## Phase 5: Training Modifications

### Files to Modify
- `train.py`

### 5.1 Progressive Loop Training (Optional)

Curriculum learning approach - start with fewer loops and gradually increase:

```python
def get_current_n_loops(step, initial_loops, target_loops, warmup_steps):
    """Gradually increase number of loops during training."""
    if step >= warmup_steps:
        return target_loops
    progress = step / warmup_steps
    return int(initial_loops + progress * (target_loops - initial_loops))

# In training loop:
if train_cfg.progressive_loops:
    current_loops = get_current_n_loops(
        step,
        initial_loops=2,
        target_loops=vlm_cfg.lm_n_loops,
        warmup_steps=5000
    )
    model.decoder.set_n_loops(current_loops)
```

Add method to `LanguageModel`:
```python
def set_n_loops(self, n_loops: int):
    """Dynamically set number of loops (for progressive training)."""
    assert n_loops <= self.cfg.lm_n_loops, "Cannot exceed max loops"
    self.current_n_loops = n_loops
```

### 5.2 Regularization

Add regularization to encourage different behavior across loops:

#### Loop Diversity Loss
Penalize hidden states being too similar across loops:
```python
def loop_diversity_loss(hidden_states_per_loop: list[torch.Tensor]) -> torch.Tensor:
    """Encourage different representations at each loop."""
    loss = 0
    for i in range(len(hidden_states_per_loop) - 1):
        # Cosine similarity between consecutive loop outputs
        h1 = hidden_states_per_loop[i].flatten(1)
        h2 = hidden_states_per_loop[i + 1].flatten(1)
        similarity = F.cosine_similarity(h1, h2, dim=1).mean()
        loss += similarity
    return loss / (len(hidden_states_per_loop) - 1)
```

### 5.3 Learning Rate Adjustments

Consider separate learning rates for different components:

```python
def get_optimizer_param_groups(model, train_cfg, vlm_cfg):
    param_groups = []

    if vlm_cfg.lm_looped:
        # Core blocks - use language backbone LR
        param_groups.append({
            'params': model.decoder.core_blocks.parameters(),
            'lr': train_cfg.lr_language_backbone,
            'name': 'lm_core_blocks'
        })
        # Loop embeddings - use higher LR (new parameters)
        param_groups.append({
            'params': model.decoder.loop_embedding.parameters(),
            'lr': train_cfg.lr_mp,  # Same as modality projector
            'name': 'lm_loop_embedding'
        })
    else:
        param_groups.append({
            'params': model.decoder.parameters(),
            'lr': train_cfg.lr_language_backbone,
            'name': 'lm_backbone'
        })

    # ... other param groups (vision, MP)
    return param_groups
```

---

## Phase 6: Advanced Features (Optional)

### 6.1 Adaptive Computation / Early Exit

Allow the model to exit early if confidence is high:

```python
class AdaptiveLoopExit(nn.Module):
    """Predicts whether to exit early based on hidden state."""
    def __init__(self, cfg):
        super().__init__()
        self.classifier = nn.Linear(cfg.lm_hidden_dim, 1)
        self.threshold = 0.9

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Returns exit probability and whether to exit."""
        logit = self.classifier(hidden_state.mean(dim=1))  # [B, 1]
        prob = torch.sigmoid(logit)
        should_exit = prob.mean() > self.threshold
        return prob, should_exit

class LoopedLanguageModel(LanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.lm_adaptive_exit:
            self.exit_predictor = AdaptiveLoopExit(cfg)

    def forward(self, x, ...):
        ...
        for loop_idx in range(self.n_loops):
            x = self.process_loop(x, loop_idx, ...)

            if self.cfg.lm_adaptive_exit and not self.training:
                _, should_exit = self.exit_predictor(x)
                if should_exit:
                    break
        ...
```

### 6.2 Per-Token Adaptive Loops (Advanced)

Different tokens may need different numbers of iterations:

```python
def adaptive_forward(self, x, ...):
    """Per-token adaptive computation."""
    B, T, D = x.shape
    halted = torch.zeros(B, T, device=x.device, dtype=torch.bool)

    for loop_idx in range(self.n_loops):
        # Only process non-halted tokens
        active_mask = ~halted
        if not active_mask.any():
            break

        x_active = x[active_mask]
        x_active = self.process_loop(x_active, loop_idx, ...)
        x[active_mask] = x_active

        # Update halting decisions
        halt_prob = self.exit_predictor(x)
        halted = halted | (halt_prob > 0.9)

    return x
```

---

## Phase 7: VLM Integration

### Files to Modify
- `models/vision_language_model.py`

### Changes Required

The `VisionLanguageModel` class should work with minimal changes if looped logic is encapsulated in `LanguageModel` and `ViT`:

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            # from_pretrained now handles looped config
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        ...
```

Ensure `generate()` method works correctly with the new KV cache structure:
- KV cache length should be `n_loops × n_core_blocks` for looped mode
- Cache indices need to map correctly to virtual layers

---

## Implementation Order & Priority

| Step | Task | Priority | Complexity | Files |
|------|------|----------|------------|-------|
| 1 | Add config parameters | High | Low | `config.py` |
| 2 | Implement `LoopPositionEmbedding` | High | Low | `language_model.py` |
| 3 | Modify `LanguageModel.__init__` | High | Medium | `language_model.py` |
| 4 | Modify `LanguageModel.forward` | High | Medium | `language_model.py` |
| 5 | Update KV cache handling | High | Medium | `language_model.py` |
| 6 | Update `LanguageModel.from_pretrained` | Medium | Medium | `language_model.py` |
| 7 | Implement `ViTLoopPositionEmbedding` | Medium | Low | `vision_transformer.py` |
| 8 | Modify `ViT` for looping | Medium | Medium | `vision_transformer.py` |
| 9 | Update `ViT.from_pretrained` | Medium | Medium | `vision_transformer.py` |
| 10 | Add training modifications | Medium | Low | `train.py` |
| 11 | Implement progressive training | Low | Low | `train.py` |
| 12 | Implement adaptive exit | Low | High | `language_model.py` |
| 13 | Add unit tests | High | Low | `tests/` |

---

## Testing Strategy

### Unit Tests to Add (`tests/`)

```python
# tests/test_looped_transformer.py

def test_looped_lm_output_shape():
    """Verify looped LM produces correct output shape."""
    cfg = VLMConfig(lm_looped=True, lm_n_core_blocks=4, lm_n_loops=8)
    model = LanguageModel(cfg)
    x = torch.randn(2, 10, cfg.lm_hidden_dim)
    out, kv_cache = model(x)
    assert out.shape == x.shape
    assert len(kv_cache) == cfg.lm_n_core_blocks * cfg.lm_n_loops

def test_looped_lm_parameter_reduction():
    """Verify looped model has fewer parameters."""
    cfg_standard = VLMConfig(lm_looped=False, lm_n_blocks=32)
    cfg_looped = VLMConfig(lm_looped=True, lm_n_core_blocks=4, lm_n_loops=8)

    model_standard = LanguageModel(cfg_standard)
    model_looped = LanguageModel(cfg_looped)

    params_standard = sum(p.numel() for p in model_standard.parameters())
    params_looped = sum(p.numel() for p in model_looped.parameters())

    assert params_looped < params_standard * 0.2  # At least 80% reduction in block params

def test_kv_cache_consistency():
    """Verify KV cache works correctly across loops."""
    cfg = VLMConfig(lm_looped=True, lm_n_core_blocks=4, lm_n_loops=8)
    model = LanguageModel(cfg)
    model.eval()

    # Prefill
    x = torch.randn(1, 10, cfg.lm_hidden_dim)
    _, kv_cache = model(x, kv_cache=None, start_pos=0)

    # Decode step
    x_new = torch.randn(1, 1, cfg.lm_hidden_dim)
    _, kv_cache = model(x_new, kv_cache=kv_cache, start_pos=10)

    # Check cache grew correctly
    for cache in kv_cache:
        assert cache['key'].shape[2] == 11  # 10 + 1

def test_generation_looped():
    """Verify generation works with looped model."""
    cfg = VLMConfig(lm_looped=True, lm_n_core_blocks=4, lm_n_loops=8, lm_use_tokens=True)
    model = LanguageModel(cfg)
    model.eval()

    inputs = torch.randint(0, cfg.lm_vocab_size, (1, 5))
    outputs = model.generate(inputs, max_new_tokens=10)

    assert outputs.shape == (1, 15)  # 5 input + 10 generated
```

### Integration Tests

1. **Full training loop**: Train looped model for 100 steps, verify loss decreases
2. **Evaluation comparison**: Compare looped vs standard model on same eval set
3. **Inference benchmarks**: Measure tokens/second for looped vs standard

---

## Expected Outcomes

### Parameter Reduction

| Configuration | Standard | Looped (4 core, 8 loops) | Reduction |
|--------------|----------|--------------------------|-----------|
| LM Block Params | 32 × ~11M = 352M | 4 × ~11M = 44M | ~87% |
| ViT Block Params | 12 × ~7M = 84M | 3 × ~7M = 21M | ~75% |
| Total Model | ~450M | ~120M | ~73% |

### Trade-offs

**Pros:**
- Significantly fewer parameters (better for deployment)
- Potentially better generalization (parameter sharing acts as regularization)
- More compute-efficient per parameter
- Easier to fit in memory

**Cons:**
- Sequential loop iterations may slow training (less parallelism)
- May require careful hyperparameter tuning
- KV cache memory similar to standard model
- Pretrained weight transfer may not be optimal

### Performance Expectations
- Initial performance may be lower than standard model with same effective depth
- With proper training, can match or exceed standard model at lower param count
- Progressive training and proper initialization are key to success

---

## References

1. **Universal Transformers** (Dehghani et al., 2018) - Original looped/iterative transformer paper
2. **ALBERT** (Lan et al., 2019) - Cross-layer parameter sharing in BERT
3. **Adaptive Computation Time** (Graves, 2016) - Dynamic halting mechanisms
4. **PonderNet** (Banino et al., 2021) - Learned adaptive computation

---

## Appendix: Quick Start Implementation

For a minimal working implementation, focus on these changes only:

1. **config.py**: Add `lm_looped`, `lm_n_core_blocks`, `lm_n_loops`
2. **language_model.py**:
   - Add `LoopPositionEmbedding` class
   - Modify `LanguageModel.__init__` to create `core_blocks` when looped
   - Modify `LanguageModel.forward` to loop over core blocks
3. **Test**: Verify shapes and basic functionality

This minimal implementation can be done in ~100 lines of code changes.
