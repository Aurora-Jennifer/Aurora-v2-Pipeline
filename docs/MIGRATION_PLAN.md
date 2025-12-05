# Documentation Migration Plan

This document tracks the migration of existing documentation into the new 4-tier architecture.

## Current Status: PLANNING

## Categorization Map

### Tier A: Executive / High-Level
**Target Location**: `docs/00_executive/`

**Files to Create/Migrate**:
- [ ] `README.md` (root) - Already exists, needs rewrite
- [ ] `QUICKSTART.md` - New, consolidate from INFORMATION/01_QUICK_START.md
- [ ] `ARCHITECTURE_OVERVIEW.md` - New, consolidate from multiple sources
- [ ] `GETTING_STARTED.md` - New, onboarding flow

**Source Files**:
- `README.md` (root)
- `INFORMATION/07_PROJECT_OVERVIEW.md`
- `IBKR_trading/README.md` (extract overview)
- `ALPACA_trading/README.md` (extract overview)

---

### Tier B: Tutorials / Walkthroughs
**Target Location**: `docs/01_tutorials/`

#### Setup Tutorials
- [ ] `setup/INSTALLATION.md` - New
- [ ] `setup/ENVIRONMENT_SETUP.md` - New
- [ ] `setup/GPU_SETUP.md` - From `docs/GPU_SETUP_MULTI_MODEL.md`, `dep/GPU_FEATURE_SELECTION_GUIDE.md`, `dep/QUICK_START_GPU.md`

#### Pipeline Tutorials
- [ ] `pipelines/FIRST_PIPELINE_RUN.md` - New
- [ ] `pipelines/DATA_PROCESSING_WALKTHROUGH.md` - From `INFORMATION/04_DATA_PIPELINE.md`
- [ ] `pipelines/FEATURE_ENGINEERING_TUTORIAL.md` - New, consolidate feature docs

#### Training Tutorials
- [ ] `training/MODEL_TRAINING_GUIDE.md` - From `INFORMATION/05_MODEL_TRAINING.md`
- [ ] `training/WALKFORWARD_VALIDATION.md` - New
- [ ] `training/FEATURE_SELECTION_TUTORIAL.md` - From `TRAINING/FEATURE_SELECTION_GUIDE.md`, `NOTES/QUICK_START_FEATURE_RANKING.md`

#### Trading Tutorials
- [ ] `trading/PAPER_TRADING_SETUP.md` - New
- [ ] `trading/IBKR_INTEGRATION.md` - From `IBKR_trading/README.md`, `IBKR_trading/LIVE_TRADING_INTEGRATION.md`
- [ ] `trading/ALPACA_INTEGRATION.md` - From `ALPACA_trading/README.md`

#### Configuration Tutorials
- [ ] `configuration/CONFIG_BASICS.md` - From `INFORMATION/01_QUICK_START.md`, `INFORMATION/02_CONFIG_REFERENCE.md`
- [ ] `configuration/CONFIG_EXAMPLES.md` - New
- [ ] `configuration/ADVANCED_CONFIG.md` - From `CONFIG/README.md`

---

### Tier C: Core Reference Docs
**Target Location**: `docs/02_reference/`

#### API Reference
- [ ] `api/MODULE_REFERENCE.md` - New, auto-generate from code
- [ ] `api/CLI_REFERENCE.md` - New, document all CLI commands
- [ ] `api/CONFIG_SCHEMA.md` - From `INFORMATION/02_CONFIG_REFERENCE.md`

#### Data Reference
- [ ] `data/DATA_FORMAT_SPEC.md` - New
- [ ] `data/COLUMN_REFERENCE.md` - From `INFORMATION/06_COLUMN_REFERENCE.md`
- [ ] `data/DATA_SANITY_RULES.md` - New, from code/config

#### Models Reference
- [ ] `models/MODEL_CATALOG.md` - New, list all 17+ models
- [ ] `models/MODEL_CONFIG_REFERENCE.md` - From `CONFIG/model_config/` files
- [ ] `models/TRAINING_PARAMETERS.md` - New

#### Systems Reference
- [ ] `systems/IBKR_SYSTEM_REFERENCE.md` - From `IBKR_trading/README.md`, `IBKR_trading/live_trading/README.md`
- [ ] `systems/ALPACA_SYSTEM_REFERENCE.md` - From `ALPACA_trading/README.md`
- [ ] `systems/PIPELINE_REFERENCE.md` - From `DATA_PROCESSING/README.md`

#### Configuration Reference
- [ ] `configuration/CONFIG_LOADER_API.md` - From `CONFIG/config_loader.py`
- [ ] `configuration/CONFIG_OVERLAYS.md` - New
- [ ] `configuration/ENVIRONMENT_VARIABLES.md` - New

---

### Tier D: Deep Technical Appendices
**Target Location**: `docs/03_technical/`

#### Research
- [ ] `research/LEAKAGE_ANALYSIS.md` - From `docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md`, `docs/TARGET_LEAKAGE_CLARIFICATION.md`, `docs/VALIDATION_LEAK_AUDIT.md`, `docs/FIXES/*.md`
- [ ] `research/FEATURE_IMPORTANCE_METHODOLOGY.md` - From `docs/FEATURE_IMPORTANCE_FIX.md`, `docs/IMPORTANCE_R2_WEIGHTING.md`, `docs/IMPORTANCE_SCORE_INTERPRETATION.md`
- [ ] `research/TARGET_DISCOVERY.md` - From `docs/TARGET_DISCOVERY_UPDATE.md`, `docs/TARGET_RECOMMENDATIONS.md`, `docs/TARGET_MODEL_PIPELINE_ANALYSIS.md`
- [ ] `research/VALIDATION_METHODOLOGY.md` - New, consolidate validation docs

#### Design
- [ ] `design/ARCHITECTURE_DEEP_DIVE.md` - New, consolidate architecture docs
- [ ] `design/MATHEMATICAL_FOUNDATIONS.md` - From `IBKR_trading/MATHEMATICAL_FOUNDATIONS.md`
- [ ] `design/OPTIMIZATION_ENGINE.md` - From `IBKR_trading/OPTIMIZATION_ENGINE_ANALYSIS.md`, `IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md`
- [ ] `design/C++_INTEGRATION.md` - From `IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md`, `IBKR_trading/cpp_engine/README.md`

#### Benchmarks
- [ ] `benchmarks/PERFORMANCE_METRICS.md` - New
- [ ] `benchmarks/MODEL_COMPARISONS.md` - From `docs/COMPLETE_FEATURE_SELECTION_MODELS.md`, `docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md`
- [ ] `benchmarks/DATASET_SIZING.md` - From `docs/DATASET_SIZING_STRATEGY.md`

#### Fixes
- [ ] `fixes/KNOWN_ISSUES.md` - From `docs/CODE_REVIEW_BUGS.md`
- [ ] `fixes/BUG_FIXES.md` - Consolidate from `docs/FIXES/*.md`
- [ ] `fixes/MIGRATION_NOTES.md` - From `INFORMATION/03_MIGRATION_NOTES.md`

#### Roadmaps
- [ ] `roadmaps/ALPHA_ENHANCEMENT_ROADMAP.md` - From `docs/ALPHA_ENHANCEMENT_ROADMAP.md`
- [ ] `roadmaps/FUTURE_WORK.md` - From `ROADMAP.md`, `NOTES/WHAT_TO_DO_NEXT.md`

#### Implementation Status
- [ ] `implementation/IBKR_STATUS.md` - From `IBKR_trading/IMPLEMENTATION_STATUS.md`
- [ ] `implementation/PRESSURE_TEST_PLAN.md` - From `IBKR_trading/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md`, `IBKR_trading/PRESSURE_TEST_UPGRADES.md`
- [ ] `implementation/PERFORMANCE_OPTIMIZATION.md` - From `IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md`

#### Testing
- [ ] `testing/TESTING_PLAN.md` - From `IBKR_trading/TESTING_PLAN.md`
- [ ] `testing/TESTING_SUMMARY.md` - From `IBKR_trading/TESTING_SUMMARY.md`
- [ ] `testing/DAILY_TESTING.md` - From `IBKR_trading/DAILY_TESTING_README.md`

#### Operations
- [ ] `operations/JOURNALD_LOGGING.md` - From `docs/JOURNALD_LOGGING.md`
- [ ] `operations/RESTORE_FROM_LOGS.md` - From `docs/RESTORE_FROM_LOGS.md`
- [ ] `operations/AVOID_LONG_RUNS.md` - From `docs/AVOID_LONG_RUNS.md`
- [ ] `operations/SYSTEMD_DEPLOYMENT.md` - From `IBKR_trading/SYSTEMD_DEPLOYMENT_PLAN.md`

---

## Files to Archive (Keep for Reference, Don't Migrate)

These files contain historical information but don't fit the new structure:

- `IBKR_trading/deprecated/*.md` - Keep in deprecated folder
- `scripts/OUTDATED_SCRIPTS.md` - Keep as-is
- `TRAINING/strategies/STRATEGY_UPDATES.md` - May archive
- `INFORMATION/*.md` - Will be replaced by new structure
- `NOTES/*.md` - May archive or consolidate

---

## Deduplication Targets

### Feature Selection (High Redundancy)
- `docs/COMPREHENSIVE_FEATURE_RANKING.md`
- `docs/COMPLETE_FEATURE_SELECTION_MODELS.md`
- `docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md`
- `docs/ADDITIONAL_MODELS_QUICKSTART.md`
- `docs/ALL_MODELS_ENABLED.md`
- `TRAINING/FEATURE_SELECTION_GUIDE.md`
- `NOTES/QUICK_START_FEATURE_RANKING.md`

**Action**: Consolidate into:
- `01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md`
- `02_reference/models/MODEL_CATALOG.md`
- `03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md`

### Leakage Documentation (High Redundancy)
- `docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md`
- `docs/TARGET_LEAKAGE_CLARIFICATION.md`
- `docs/VALIDATION_LEAK_AUDIT.md`
- `docs/FIXES/DEEPER_LEAK_FIX.md`
- `docs/FIXES/FINAL_LEAKAGE_SUMMARY.md`
- `docs/FIXES/LEAKAGE_FIXED_NEXT_STEPS.md`
- `docs/FIXES/ROUND3_TEMPORAL_OVERLAP_FIX.md`
- `docs/FIXES/TARGET_IS_LEAKED.md`
- `dep/LEAKAGE_FIX_README.md`

**Action**: Consolidate into:
- `03_technical/research/LEAKAGE_ANALYSIS.md` (comprehensive)
- `03_technical/fixes/BUG_FIXES.md` (fix history)

### IBKR Integration (Some Redundancy)
- `IBKR_trading/README.md`
- `IBKR_trading/LIVE_TRADING_INTEGRATION.md`
- `IBKR_trading/live_trading/README.md`
- `IBKR_trading/ENHANCED_REBALANCING_TRADING_PLAN.md`
- `IBKR_trading/INTRADAY_TRADING_ANALYSIS.md`

**Action**: Split into:
- `01_tutorials/trading/IBKR_INTEGRATION.md` (tutorial)
- `02_reference/systems/IBKR_SYSTEM_REFERENCE.md` (reference)

---

## Migration Phases

### Phase 1: Structure Creation (Current)
- [x] Create architecture document
- [x] Create migration plan
- [ ] Create directory structure
- [ ] Create INDEX.md

### Phase 2: Entry Points (Priority)
- [ ] Rewrite root README.md
- [ ] Create QUICKSTART.md
- [ ] Create ARCHITECTURE_OVERVIEW.md
- [ ] Create GETTING_STARTED.md

### Phase 3: Tutorials
- [ ] Setup tutorials
- [ ] Pipeline tutorials
- [ ] Training tutorials
- [ ] Trading tutorials
- [ ] Configuration tutorials

### Phase 4: Reference Docs
- [ ] API reference
- [ ] Data reference
- [ ] Models reference
- [ ] Systems reference
- [ ] Configuration reference

### Phase 5: Technical Appendices
- [ ] Research docs
- [ ] Design docs
- [ ] Benchmarks
- [ ] Fixes
- [ ] Roadmaps

### Phase 6: Cleanup
- [ ] Remove redundant files
- [ ] Add cross-links
- [ ] Verify all links work
- [ ] Create search index

---

## Next Steps

1. Review and approve this plan
2. Create directory structure
3. Start with Phase 2 (entry points) - highest impact
4. Migrate tutorials (Phase 3) - high user value
5. Complete reference docs (Phase 4) - daily use
6. Archive technical appendices (Phase 5) - deep dive

