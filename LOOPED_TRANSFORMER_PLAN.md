# Looped Transformer Implementation Plan for nanoVLM

## Overview

A **looped transformer** reuses a single transformer block (or small set of blocks) multiple times, dramatically reducing parameters while maintaining effective depth.

### Architecture: Prelude → Looped Core → Coda

```
Input Embeddings
       ↓
┌─────────────────┐
│  PRELUDE LAYERS │  ← Unique weights (N layers)
│  (not shared)   │
└─────────────────┘
       ↓
┌─────────────────┐
│   CORE LAYERS   │  ← Shared weights (K layers × L loops)
│    (looped)     │  ← Same K blocks repeated L times
└─────────────────┘
       ↓
┌─────────────────┐
│   CODA LAYERS   │  ← Unique weights (M layers)
│  (not shared)   │
└─────────────────┘
       ↓
   Output / Head
```

**Why prelude/coda?**
- **Prelude layers**: Process raw embeddings into a representation suitable for iterative refinement
- **Coda layers**: Transform the refined representation into output-ready features
- **Looped core**: Performs iterative refinement with shared weights

### Example Configuration

| Component | Layers | Parameters |
|-----------|--------|------------|
| Prelude | 2 unique layers | 2 × ~11M = 22M |
| Core | 4 layers × 7 loops = 28 effective | 4 × ~11M = 44M |
| Coda | 2 unique layers | 2 × ~11M = 22M |
| **Total** | **32 effective depth** | **88M** (vs 352M standard) |

This gives **75% parameter reduction** in the transformer blocks while maintaining the same effective depth.

---

## Step 1: Basic Looped Transformer (Minimal Implementation)

This is the simplest possible implementation with no bells and whistles.

### 1.1 Configuration Changes

**File: `models/config.py`**

Add these parameters to `VLMConfig`:

```python
# Looped Transformer Configuration (Language Model only)
lm_looped: bool = False              # Enable looped transformer
lm_n_prelude_layers: int = 2         # Number of unique prelude layers
lm_n_core_layers: int = 4            # Number of unique core layers (to be looped)
lm_n_loops: int = 7                  # Number of times to loop the core layers
lm_n_coda_layers: int = 2            # Number of unique coda layers
# Effective depth = prelude + (core × loops) + coda = 2 + (4 × 7) + 2 = 32
```

### 1.2 Language Model Changes

**File: `models/language_model.py`**

#### Modify `LanguageModel.__init__`:

```python
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)

        # Looped vs Standard architecture
        if getattr(cfg, 'lm_looped', False):
            self.looped = True
            self.n_loops = cfg.lm_n_loops

            # Prelude: unique layers at the start
            self.prelude_blocks = nn.ModuleList([
                LanguageModelBlock(cfg) for _ in range(cfg.lm_n_prelude_layers)
            ])

            # Core: shared layers that get looped
            self.core_blocks = nn.ModuleList([
                LanguageModelBlock(cfg) for _ in range(cfg.lm_n_core_layers)
            ])

            # Coda: unique layers at the end
            self.coda_blocks = nn.ModuleList([
                LanguageModelBlock(cfg) for _ in range(cfg.lm_n_coda_layers)
            ])

            # Total virtual layers for KV cache
            self.n_virtual_layers = (
                cfg.lm_n_prelude_layers +
                cfg.lm_n_core_layers * cfg.lm_n_loops +
                cfg.lm_n_coda_layers
            )
        else:
            self.looped = False
            self.blocks = nn.ModuleList([
                LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
            ])

        self.norm = RMSNorm(cfg)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
```

#### Modify `LanguageModel.forward`:

```python
def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None,
            kv_cache: list[dict]=None, start_pos: int=0):
    if self.lm_use_tokens:
        x = self.token_embedding(x)

    B, T_curr, _ = x.size()
    current_position_ids = torch.arange(start_pos, start_pos + T_curr,
                                         device=x.device).unsqueeze(0).expand(B, -1)
    cos, sin = self.rotary_embd(current_position_ids)

    if self.looped:
        # Initialize KV cache for all virtual layers
        if kv_cache is None:
            kv_cache = [None] * self.n_virtual_layers

        cache_idx = 0

        # Prelude layers (unique, not looped)
        for block in self.prelude_blocks:
            x, kv_cache[cache_idx] = block(x, cos, sin, attention_mask, kv_cache[cache_idx])
            cache_idx += 1

        # Core layers (looped)
        for loop_idx in range(self.n_loops):
            for block in self.core_blocks:
                x, kv_cache[cache_idx] = block(x, cos, sin, attention_mask, kv_cache[cache_idx])
                cache_idx += 1

        # Coda layers (unique, not looped)
        for block in self.coda_blocks:
            x, kv_cache[cache_idx] = block(x, cos, sin, attention_mask, kv_cache[cache_idx])
            cache_idx += 1
    else:
        # Standard (non-looped) forward pass
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

    x = self.norm(x)

    if self.lm_use_tokens:
        x = self.head(x)

    return x, kv_cache
```

### 1.3 Pretrained Weight Loading

**File: `models/language_model.py`** - Modify `from_pretrained`

For the basic implementation, use a simple strategy: map pretrained layers directly to prelude/core/coda.

```python
@classmethod
def from_pretrained(cls, cfg):
    # ... existing HF config loading code ...

    # Create model
    model = cls(cfg)

    # ... existing safetensors loading code ...

    if getattr(cfg, 'lm_looped', False):
        # Map pretrained layers to looped architecture
        # Strategy: First N → prelude, next K → core, last M → coda

        n_pretrained = cfg.lm_n_blocks  # e.g., 32 from SmolLM2
        n_prelude = cfg.lm_n_prelude_layers
        n_core = cfg.lm_n_core_layers
        n_coda = cfg.lm_n_coda_layers

        # Load prelude from first N pretrained layers
        for i in range(n_prelude):
            src_prefix = f'blocks.{i}.'
            dst_prefix = f'prelude_blocks.{i}.'
            copy_block_weights(sd, src_prefix, dst_prefix)

        # Load core from middle layers (evenly spaced)
        core_start = n_prelude
        core_spacing = (n_pretrained - n_prelude - n_coda) // n_core
        for i in range(n_core):
            src_idx = core_start + i * core_spacing
            src_prefix = f'blocks.{src_idx}.'
            dst_prefix = f'core_blocks.{i}.'
            copy_block_weights(sd, src_prefix, dst_prefix)

        # Load coda from last M pretrained layers
        for i in range(n_coda):
            src_idx = n_pretrained - n_coda + i
            src_prefix = f'blocks.{src_idx}.'
            dst_prefix = f'coda_blocks.{i}.'
            copy_block_weights(sd, src_prefix, dst_prefix)
    else:
        # Standard loading (existing code)
        ...
```

### 1.4 Summary of Step 1 Changes

| File | Changes |
|------|---------|
| `models/config.py` | Add 5 new config parameters |
| `models/language_model.py` | Modify `__init__`, `forward`, `from_pretrained` |

**Lines of code to change**: ~100-150 lines

**No changes needed to**:
- `vision_transformer.py` (ViT stays unchanged)
- `vision_language_model.py` (VLM uses decoder interface unchanged)
- `modality_projector.py`
- `train.py` (training loop unchanged)

---

## Step 2: Testing the Basic Implementation

### 2.1 Unit Tests

**File: `tests/test_looped_transformer.py`**

```python
import torch
from models.config import VLMConfig
from models.language_model import LanguageModel

def test_looped_lm_output_shape():
    """Verify looped LM produces correct output shape."""
    cfg = VLMConfig(
        lm_looped=True,
        lm_n_prelude_layers=2,
        lm_n_core_layers=4,
        lm_n_loops=7,
        lm_n_coda_layers=2
    )
    model = LanguageModel(cfg)
    x = torch.randn(2, 10, cfg.lm_hidden_dim)
    out, kv_cache = model(x)

    assert out.shape == x.shape
    # 2 + 4*7 + 2 = 32 virtual layers
    assert len(kv_cache) == 32

def test_looped_lm_parameter_count():
    """Verify parameter reduction."""
    cfg_standard = VLMConfig(lm_looped=False, lm_n_blocks=32)
    cfg_looped = VLMConfig(
        lm_looped=True,
        lm_n_prelude_layers=2,
        lm_n_core_layers=4,
        lm_n_loops=7,
        lm_n_coda_layers=2
    )

    model_standard = LanguageModel(cfg_standard)
    model_looped = LanguageModel(cfg_looped)

    # Count only block parameters (excluding embeddings, norm, head)
    def count_block_params(model):
        if hasattr(model, 'blocks'):
            return sum(p.numel() for p in model.blocks.parameters())
        else:
            return (
                sum(p.numel() for p in model.prelude_blocks.parameters()) +
                sum(p.numel() for p in model.core_blocks.parameters()) +
                sum(p.numel() for p in model.coda_blocks.parameters())
            )

    standard_params = count_block_params(model_standard)
    looped_params = count_block_params(model_looped)

    # Looped should have 8/32 = 25% of standard block params
    assert looped_params < standard_params * 0.3

def test_kv_cache_generation():
    """Verify KV cache works during generation."""
    cfg = VLMConfig(
        lm_looped=True,
        lm_n_prelude_layers=2,
        lm_n_core_layers=4,
        lm_n_loops=7,
        lm_n_coda_layers=2,
        lm_use_tokens=True
    )
    model = LanguageModel(cfg)
    model.eval()

    # Prefill
    input_ids = torch.randint(0, 1000, (1, 5))
    _, kv_cache = model(input_ids, kv_cache=None, start_pos=0)

    # Decode one token
    new_token = torch.randint(0, 1000, (1, 1))
    _, kv_cache = model(new_token, kv_cache=kv_cache, start_pos=5)

    # Check cache grew
    assert kv_cache[0]['key'].shape[2] == 6  # 5 + 1
```

---

## Step 3: Advanced Features (Optional Extensions)

Once the basic implementation works, these features can be added incrementally.

### 3.1 Loop Position Embedding

Inject information about which loop iteration is being processed.

```python
class LoopPositionEmbedding(nn.Module):
    """Additive embedding to indicate loop iteration."""
    def __init__(self, n_loops, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_loops, hidden_dim)

    def forward(self, x, loop_idx):
        # Add loop embedding to all positions
        loop_emb = self.embedding(torch.tensor([loop_idx], device=x.device))
        return x + loop_emb.unsqueeze(1)  # [1, 1, hidden_dim] broadcasts

# In forward:
for loop_idx in range(self.n_loops):
    x = self.loop_embedding(x, loop_idx)  # Add before each loop
    for block in self.core_blocks:
        x, kv_cache[cache_idx] = block(...)
```

### 3.2 Progressive Loop Training

Start with fewer loops and gradually increase during training.

```python
# In train.py
def get_n_loops(step, min_loops=2, max_loops=7, warmup_steps=5000):
    if step >= warmup_steps:
        return max_loops
    return min_loops + int((max_loops - min_loops) * step / warmup_steps)

# During training
current_loops = get_n_loops(step)
model.decoder.set_n_loops(current_loops)
```

### 3.3 Adaptive Early Exit

Allow the model to exit the loop early based on confidence.

```python
class EarlyExitClassifier(nn.Module):
    def __init__(self, hidden_dim, threshold=0.9):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)
        self.threshold = threshold

    def should_exit(self, x):
        # Only during inference
        logit = self.classifier(x.mean(dim=1))  # [B, 1]
        return torch.sigmoid(logit).mean() > self.threshold

# In forward (inference only):
for loop_idx in range(self.n_loops):
    for block in self.core_blocks:
        x, kv_cache[cache_idx] = block(...)

    if not self.training and self.early_exit.should_exit(x):
        break
```

### 3.4 Dense Residual Connections

Add skip connections from input to each loop iteration.

```python
# Store input to core
core_input = x.clone()

for loop_idx in range(self.n_loops):
    for block in self.core_blocks:
        x, kv_cache[cache_idx] = block(...)
    # Add residual from core input
    x = x + core_input * 0.1  # Small weight to not dominate
```

---

## Implementation Checklist

### Phase 1: Basic Implementation
- [ ] Add config parameters to `VLMConfig`
- [ ] Modify `LanguageModel.__init__` for prelude/core/coda
- [ ] Modify `LanguageModel.forward` for looped execution
- [ ] Update `from_pretrained` for weight loading
- [ ] Write unit tests
- [ ] Verify generation works with KV cache

### Phase 2: Training & Validation
- [ ] Train basic looped model
- [ ] Compare performance vs standard model
- [ ] Benchmark inference speed
- [ ] Measure memory usage

### Phase 3: Optional Enhancements (pick as needed)
- [ ] Loop position embedding
- [ ] Progressive loop training
- [ ] Adaptive early exit
- [ ] Dense residual connections

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ViT looping | No | Keep vision encoder standard; focus on LM |
| Prelude/Coda | Yes | Allows unique input/output processing |
| Loop embedding | No (basic) | Start simple, add if needed |
| KV cache strategy | Separate per virtual layer | Maintains effective depth, compatible with existing code |
| Weight init | Map from pretrained | Faster convergence than random init |

---

## Expected Results

### Parameter Count (SmolLM2-360M equivalent depth)

| Architecture | Block Params | Total Params | Reduction |
|--------------|--------------|--------------|-----------|
| Standard (32 blocks) | ~352M | ~360M | - |
| Looped (2+4+2 blocks) | ~88M | ~96M | 75% |

### Trade-offs

**Pros:**
- 75% fewer parameters in transformer blocks
- Same effective depth (32 layers)
- Potentially better generalization
- Faster to load/save

**Cons:**
- Sequential loops reduce parallelism
- May need more training iterations
- KV cache memory unchanged
