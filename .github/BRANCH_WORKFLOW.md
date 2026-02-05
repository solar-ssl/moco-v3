# MoCo v3 Fix Branch Workflow

## Quick Reference

### Created Fix Branches (All from main)

```
main
├── fix/projection-prediction-head-batchnorm     [b47290d] P0 - CRITICAL
├── fix/correct-batch-size-lr-schedule            [38a1420] P0 - CRITICAL  
├── fix/queue-update-logic                        [a95d00b] P1 - MODERATE-HIGH
├── fix/satellite-augmentations                   [8ec056d] P2 - MODERATE
└── fix/vit-patch-projection-freeze              [e0470ea] P0 - CRITICAL
```

### Merging Strategy

#### Option 1: Sequential (Recommended)
```bash
git checkout main

# Merge in priority order
git merge fix/projection-prediction-head-batchnorm
git merge fix/correct-batch-size-lr-schedule
git merge fix/queue-update-logic
git merge fix/satellite-augmentations
git merge fix/vit-patch-projection-freeze

# Test after each merge
python -m src.utils.verify_training
```

#### Option 2: All at once
```bash
git checkout main
git merge fix/projection-prediction-head-batchnorm \
          fix/correct-batch-size-lr-schedule \
          fix/queue-update-logic \
          fix/satellite-augmentations \
          fix/vit-patch-projection-freeze
```

#### Option 3: Combined feature branch
```bash
git checkout -b feature/moco-v3-paper-compliance main
git cherry-pick b47290d  # BN fix
git cherry-pick 38a1420  # Batch size fix
git cherry-pick a95d00b  # Queue fix
git cherry-pick 8ec056d  # Augmentation fix
git cherry-pick e0470ea  # Patch freeze fix
```

### Testing Individual Branches

```bash
# Test BN fix
git checkout fix/projection-prediction-head-batchnorm
python -c "from src.models.moco_v3 import MoCoV3; from src.models.backbones import get_backbone; print('✓ BN fix works')"

# Test batch size fix  
git checkout fix/correct-batch-size-lr-schedule
python -c "from src.config import Config; c=Config(); assert c.batch_size==4096; print('✓ Batch size fixed')"

# Test all
git checkout main
# After merging all branches
python -m src.utils.verify_training
```

### Conflict Resolution

If merge conflicts occur (unlikely since each branch touches different files):

```bash
git status  # See conflicted files
git diff    # Review conflicts
# Manually resolve, then:
git add <resolved-files>
git commit
```

### Rollback Strategy

If a merge causes issues:

```bash
# Undo last merge
git reset --hard HEAD~1

# Or undo specific merge
git revert -m 1 <merge-commit-hash>
```

### Clean Up After Merging

```bash
# Delete merged branches (local only)
git branch -d fix/projection-prediction-head-batchnorm
git branch -d fix/correct-batch-size-lr-schedule
git branch -d fix/queue-update-logic
git branch -d fix/satellite-augmentations
git branch -d fix/vit-patch-projection-freeze
```

### Files Modified by Branch

| Branch | Files |
|--------|-------|
| fix/projection-prediction-head-batchnorm | `src/models/moco_v3.py` |
| fix/correct-batch-size-lr-schedule | `src/config.py` |
| fix/queue-update-logic | `src/models/moco_v3.py` |
| fix/satellite-augmentations | `src/utils/augmentations.py` |
| fix/vit-patch-projection-freeze | `src/models/backbones.py` |

**Note:** Only `moco_v3.py` is touched by 2 branches (BN fix + queue fix), but they modify different sections.

### Verification Checklist

After merging all fixes:

- [ ] `python -m src.utils.verify_training` passes
- [ ] Config shows batch_size=4096, epochs=300, lr=2.4e-3
- [ ] Model instantiation prints "✓ Froze ViT patch projection layer"
- [ ] Projection/prediction heads have BN on all layers
- [ ] Queue disabled by default (use_queue=False)
- [ ] Augmentations use DiscreteRotation
- [ ] All tests pass (if any exist)

### Push to Remote

```bash
# After merging and testing
git push origin main

# Or keep branches for review
git push origin fix/projection-prediction-head-batchnorm
git push origin fix/correct-batch-size-lr-schedule
# ... etc
```
