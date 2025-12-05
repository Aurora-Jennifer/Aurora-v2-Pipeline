# Documentation Cleanup Recommendations

## Summary

50+ operational documentation files have been rewritten to enterprise-grade standards. This document identifies files that should be excluded or archived for enterprise readiness.

## Files to Exclude from Enterprise Documentation

### Personal/Development Notes (Exclude)

These files contain personal notes or development artifacts that should not be in enterprise documentation:

- `NOTES/journallog.md` - Personal log file with journalctl commands
- `NOTES/WHAT_TO_DO_NEXT.md` - Personal action items (already rewritten but may be too operational for enterprise docs)

**Recommendation**: Move to `.gitignore` or archive in `docs/internal/`

### Internal Development/Fix Documentation (Archive)

These files document internal fixes and development processes:

- `docs/FIXES/*` - All files in FIXES directory (7 files)
  - Internal fix documentation
  - Should be archived or moved to `docs/internal/fixes/`
- `docs/CODE_REVIEW_BUGS.md` - Internal code review notes
- `docs/AVOID_LONG_RUNS.md` - Operational tips (may be too informal)
- `docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md` - Internal analysis
- `docs/TARGET_LEAKAGE_CLARIFICATION.md` - Internal analysis
- `docs/VALIDATION_LEAK_AUDIT.md` - Internal audit
- `docs/LEAKAGE_FIXED_NEXT_STEPS.md` - Internal notes

**Recommendation**: Move to `docs/internal/` or archive

### Development/Research Notes (Consider Archiving)

These files contain research notes and development documentation:

- `docs/TARGET_DISCOVERY_UPDATE.md` - Research notes
- `docs/TARGET_MODEL_PIPELINE_ANALYSIS.md` - Research notes
- `docs/TARGET_RECOMMENDATIONS.md` - Research notes
- `docs/TARGET_TO_FEATURE_WORKFLOW.md` - Research notes
- `docs/FEATURE_IMPORTANCE_FIX.md` - Development notes
- `docs/IMPORTANCE_R2_WEIGHTING.md` - Development notes
- `docs/IMPORTANCE_SCORE_INTERPRETATION.md` - Development notes
- `docs/DATASET_SIZING_STRATEGY.md` - Research notes
- `docs/GPU_SETUP_MULTI_MODEL.md` - Setup notes
- `docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md` - Development notes
- `docs/ADDITIONAL_MODELS_QUICKSTART.md` - Development notes
- `docs/ALL_MODELS_ENABLED.md` - Development notes
- `docs/COMPLETE_FEATURE_SELECTION_MODELS.md` - Development notes
- `docs/MODEL_ENABLING_RECOMMENDATIONS.md` - Development notes

**Recommendation**: Move to `docs/internal/research/` or archive

### Potentially Too Technical/Internal (Review)

These files may be too technical or internal for enterprise documentation:

- `docs/ALPHA_ENHANCEMENT_ROADMAP.md` - Internal roadmap
- `IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md` - Deep technical
- `IBKR_trading/INTRADAY_TRADING_ANALYSIS.md` - Internal analysis
- `IBKR_trading/OPTIMIZATION_ENGINE_ANALYSIS.md` - Internal analysis
- `IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md` - Internal planning
- `IBKR_trading/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md` - Internal roadmap
- `IBKR_trading/PRESSURE_TEST_UPGRADES.md` - Internal notes
- `IBKR_trading/ENHANCED_REBALANCING_TRADING_PLAN.md` - Internal planning
- `IBKR_trading/YAHOO_FINANCE_INTEGRATION.md` - May be outdated
- `IBKR_trading/SYSTEMD_DEPLOYMENT_PLAN.md` - Internal planning
- `IBKR_trading/MATHEMATICAL_FOUNDATIONS.md` - Deep technical (may be appropriate for technical appendices)
- `scripts/OUTDATED_SCRIPTS.md` - Should be removed
- `scripts/ranking.md` - May be outdated

**Recommendation**: Review each file - some may be appropriate for technical appendices, others should be archived

## Recommended Actions

### 1. Create Internal Documentation Directory

```bash
mkdir -p docs/internal/{fixes,research,planning}
```

### 2. Move Internal Files

```bash
# Move FIXES directory
mv docs/FIXES docs/internal/fixes/

# Move research notes
mv docs/TARGET_*.md docs/internal/research/
mv docs/FEATURE_IMPORTANCE_*.md docs/internal/research/
mv docs/IMPORTANCE_*.md docs/internal/research/
mv docs/DATASET_SIZING_STRATEGY.md docs/internal/research/
mv docs/ADDITIONAL_*.md docs/internal/research/
mv docs/ALL_MODELS_ENABLED.md docs/internal/research/
mv docs/COMPLETE_FEATURE_SELECTION_MODELS.md docs/internal/research/
mv docs/MODEL_ENABLING_RECOMMENDATIONS.md docs/internal/research/

# Move planning documents
mv docs/ALPHA_ENHANCEMENT_ROADMAP.md docs/internal/planning/
mv IBKR_trading/*PLAN*.md docs/internal/planning/
mv IBKR_trading/*ROADMAP*.md docs/internal/planning/
```

### 3. Exclude Personal Notes

Add to `.gitignore`:
```
NOTES/journallog.md
```

Or move to internal:
```bash
mv NOTES/journallog.md docs/internal/
```

### 4. Remove Outdated Files

```bash
rm scripts/OUTDATED_SCRIPTS.md
# Review scripts/ranking.md - remove if outdated
```

## Files That Should Remain (Enterprise Documentation)

These files have been rewritten and are appropriate for enterprise documentation:

### Core Documentation
- All files in `docs/00_executive/`
- `docs/INDEX.md`
- `docs/STYLE_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/MIGRATION_PLAN.md`
- `docs/RESTRUCTURING_SUMMARY.md`

### Operational Guides
- `INFORMATION/*.md` - All 8 files
- `TRAINING/*.md` - All training guides
- `ALPACA_trading/**/README.md` - All component docs
- `IBKR_trading/README.md` and core component docs
- `CONFIG/README.md`
- `DATA_PROCESSING/README.md`
- `data/README.md`

### Reference Documentation
- `docs/COMPREHENSIVE_FEATURE_RANKING.md`
- `docs/NEXT_STEPS_WORKFLOW.md`
- `docs/JOURNALD_LOGGING.md`
- `docs/RESTORE_FROM_LOGS.md`

## Summary Statistics

- **Total documentation files**: ~100
- **Rewritten to enterprise standard**: 50+
- **Should be excluded/archived**: ~30
- **Should remain (enterprise docs)**: ~70

## Next Steps

1. Review the files listed above
2. Move internal documentation to `docs/internal/`
3. Update `.gitignore` to exclude personal notes
4. Remove outdated files
5. Update `docs/INDEX.md` to reflect new structure

