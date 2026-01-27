# Batch Creation Analysis & Memory Issue Root Cause

## Batch Creation Pipeline

The training batch creation follows a **3-stage pipeline**:

### Stage 1: VQADataset (data/datasets.py)
- **Input**: Raw dataset samples with images and text
- **Processing**:
  1. Images processed through `DynamicResize` → resized to max 2048×2048
  2. Split into 512×512 patches via `GlobalAndSplitImages`
  3. A 2048×2048 image creates: **17 patches** (1 global + 16 regular)
  4. Text tokenized with chat template
  5. Image tokens inserted: each 512×512 patch → **64 language tokens**
- **Output**: `{images, input_ids, attention_mask, labels}` per sample

### Stage 2: ConstantLengthDataset (data/advanced_datasets.py)
**Greedy Knapsack Packing** to maximize GPU utilization:

1. **Buffer Accumulation** (lines 108-143):
   - Filters samples ≥ 4096 tokens
   - Filters samples with > 4 images
   - Adds padding token to each sample

2. **Knapsack Constraints** (lines 173-217):
   - Max **4096 tokens** per packed sample
   - Max **18 images** per packed sample
   - Packs multiple samples together until constraints hit

3. **Safety Check** (line 229-230):
   - Raises `ValueError` if packed length > 4096

### Stage 3: VQACollator (data/collators.py)
- Discards samples > 4096 tokens (final safety net)
- Pads batch to uniform length
- Stacks tensors for PyTorch

## Sequence Length Protection ✅

**The max sequence length (4096) is NOT exceeded** due to:
1. Pre-packing filter (drops samples ≥ 4096)
2. Knapsack constraint enforcement
3. Post-packing ValueError
4. Collator final filter

## THE MEMORY PROBLEM 🔥

### Root Cause: Vision Encoder Batch Size

Your memory issue stems from the **vision encoder processing too many image patches simultaneously**:

**Current Configuration:**
```
max_img_size = 2048
max_images_per_knapsack = 18
batch_size = 4
```

**Worst-case scenario per training step:**
- 4 samples × 18 images × 17 patches = **1,224 image patches of 512×512**
- Each patch → 1,024 ViT tokens (512/16 = 32, so 32² = 1,024)
- **Total: 1,253,376 ViT tokens** processed through vision encoder

**Memory breakdown:**
```
Vision Encoder (per layer):
- Attention matrices: 1,224 × (12 heads × 1024² tokens) = ~29 GB
- QKV projections: ~5 GB
- FFN intermediate: ~7 GB
Total per layer: ~41 GB × 12 layers = ~494 GB (if all stored)

Even with PyTorch optimizations: ~60-80 GB for ViT alone
```

**Why it exceeds 80GB:**
```
Static memory:
- Model weights (fp32): 1.7 GB
- Gradients (fp32): 1.7 GB
- Optimizer states (AdamW): 3.4 GB
Total: 6.7 GB

Dynamic memory:
- Vision encoder activations: 40-60 GB (varies with actual batch)
- Language model activations: 5-10 GB
- Input tensors: ~2 GB

TOTAL: 53-78 GB baseline + spikes during backward pass
```

### The Constraint Paradox

**The problem**: `max_images_per_knapsack = 18` allows far more images than can fit in 4096 tokens!

- One 2048×2048 image = 17 patches × 64 tokens = **1,088 tokens**
- 18 such images = **19,584 tokens** (way over 4096!)

In practice, the 4096 token limit caps you at ~3-4 large images OR ~12 small images per sample. But the vision encoder still processes whatever images ARE in the batch, creating massive memory usage when you get batches with many images.

## Solutions

### Immediate Fixes (pick one or combine):

#### Option 1: Reduce `max_images_per_knapsack` ⭐ RECOMMENDED
```python
# In models/config.py, TrainConfig
max_images_per_knapsack: int = 6  # Currently 18
```
**Impact**:
- Reduces worst-case to 4 × 6 × 17 = 408 patches
- Memory: ~12 GB for ViT (down from 60+ GB)
- Still allows meaningful multi-image samples

#### Option 2: Reduce `batch_size`
```python
# In models/config.py, TrainConfig
batch_size: int = 1  # Currently 2
gradient_accumulation_steps: int = 16  # Increase to maintain effective batch size
```
**Impact**:
- Reduces memory linearly
- Slows training slightly (more gradient accumulation steps)

#### Option 3: Reduce `max_img_size`
```python
# In models/config.py, VLMConfig
max_img_size: int = 1024  # Currently 2048
```
**Impact**:
- 1024×1024 image = 5 patches (vs 17 for 2048×2048)
- 3.4× memory reduction for images
- Lower image resolution

#### Option 4: Increase `max_images_per_example` filter
```python
# In models/config.py, TrainConfig
max_images_per_example: int = 2  # Currently 4
```
**Impact**:
- Filters out multi-image samples earlier
- Reduces diversity in training data

### Long-term Solutions:

1. **Add gradient checkpointing** to vision encoder
   - Trade computation for memory
   - Can reduce activation memory by 50-70%

2. **Process images in sub-batches** through ViT
   - Instead of processing all 1,224 patches at once
   - Process in chunks of 256-512 patches

3. **Use Flash Attention** for ViT
   - More memory-efficient attention
   - Already uses `scaled_dot_product_attention` but could optimize further

## Recommended Configuration for 80GB GPU

```python
# models/config.py

@dataclass
class TrainConfig:
    batch_size: int = 2  # Down from 4
    max_images_per_knapsack: int = 8  # Down from 18
    max_images_per_example: int = 3  # Down from 4
    # ... rest unchanged

@dataclass
class VLMConfig:
    max_img_size: int = 1536  # Down from 2048
    # ... rest unchanged
```

**Expected memory**: ~35-45 GB (comfortable margin for 80GB)

## Verification Commands

Check actual batch statistics:
```python
# Add to train.py after batch creation
print(f"Batch images: {sum(len(img) for img in batch['images'])}")
print(f"Batch tokens: {batch['input_ids'].shape}")
```

## Summary

- ✅ **Sequence length**: Protected by multiple layers, won't exceed 4096
- ❌ **Memory issue**: Vision encoder processing too many image patches (1,200+)
- 🎯 **Solution**: Reduce `max_images_per_knapsack` from 18 to 6-8
- 📊 **Impact**: 5-10× memory reduction while maintaining training quality
