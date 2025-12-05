# Documentation Style Guide

Consistent style and formatting for Fox-v1-infra documentation.

## Voice and Tone

### Voice: Active
**Good**: "The system processes data"
**Bad**: "Data is processed by the system"

### Tone: Precise, Minimal, Engineer-to-Engineer
- Be direct and clear
- Avoid marketing language
- Assume technical competence
- Explain why, not just what

### Tense: Present
**Good**: "The model trains on labeled data"
**Bad**: "The model will train on labeled data" (unless describing future plans)

## Formatting

### Headers
- Use `#` for main title (one per document)
- Use `##` for major sections
- Use `###` for subsections
- Use `####` for sub-subsections (sparingly)

### Code Blocks
- Always specify language: ` ```python`, ` ```bash`, etc.
- Include working examples
- Add comments for clarity
- Show expected output when helpful

**Example**:
```python
# Load config
from CONFIG.config_loader import load_model_config
config = load_model_config("lightgbm", variant="conservative")
# Returns: dict with model parameters
```

### Lists
- Use bullet points (`-`) for unordered lists
- Use numbered lists (`1.`) for sequential steps
- Keep items parallel in structure
- Use sub-bullets for nested information

### Emphasis
- **Bold** for important terms, file names, commands
- *Italic* for emphasis (sparingly)
- `Code` for code, file paths, config keys

### Links
- Use descriptive link text: `[Model Training Guide](path)` not `[here](path)`
- Always include "See Also" section at end of documents
- Link to related docs in same tier and other tiers

## Structure

### Document Template

```markdown
# Document Title

Brief one-sentence description.

## Overview

What this document covers and who it's for.

## Main Content

Organized sections with clear headers.

## Examples

Working code examples.

## Next Steps

What to do after reading this.

## See Also

- [Related Doc 1](path)
- [Related Doc 2](path)
```

### Section Organization

1. **Overview/Introduction** - What and why
2. **Main Content** - Core information
3. **Examples** - Practical usage
4. **Troubleshooting** - Common issues (if applicable)
5. **Next Steps** - What's next
6. **See Also** - Related docs

## Content Guidelines

### Code Examples
- Always include complete, runnable examples
- Show imports
- Include expected output when helpful
- Add comments for clarity
- Use realistic data/file names

### Configuration Examples
- Show full config structure
- Explain key parameters
- Include comments in YAML
- Show multiple variants when relevant

### File Paths
- Use relative paths from repo root
- Be consistent: `CONFIG/model_config/lightgbm.yaml`
- Use backticks: `` `CONFIG/model_config/` ``

### Terminology
- **Model**: A trained ML model
- **Config**: Configuration file (YAML)
- **Pipeline**: Data processing workflow
- **Feature**: Input variable for models
- **Target**: Output variable to predict
- **Walk-forward**: Time-series validation method

### Abbreviations
- Spell out on first use: "Walk-Forward Validation (WFV)"
- Use consistently after first use
- Common abbreviations: ML, API, CLI, GPU, CPU, OHLCV

## Cross-Referencing

### Within Document
- Use `## Section Name` for major sections
- Use `### Subsection` for subsections
- Link to sections: `[Section Name](#section-name)`

### Between Documents
- Link to related docs in same tier
- Link to reference docs for details
- Link to tutorials for step-by-step guides
- Link to technical docs for deep dives

### "See Also" Section
Every document should end with:
```markdown
## See Also

- [Related Tutorial](../01_tutorials/path)
- [Reference Doc](../02_reference/path)
- [Technical Deep Dive](../03_technical/path)
```

## Common Patterns

### Installation Instructions
```markdown
## Installation

### Prerequisites
- Python 3.11+
- 8GB+ RAM

### Steps
1. Clone repository
2. Create environment
3. Install dependencies
4. Verify installation
```

### Configuration Examples
```markdown
## Configuration

### Basic Usage
```python
from CONFIG.config_loader import load_model_config
config = load_model_config("lightgbm")
```

### Advanced Usage
```python
config = load_model_config("lightgbm", variant="conservative", overrides={"n_estimators": 2000})
```
```

### Troubleshooting
```markdown
## Troubleshooting

**Problem**: Import errors  
**Solution**: Ensure conda environment is activated

**Problem**: Out of memory  
**Solution**: Use streaming builder or reduce batch size
```

## Tier-Specific Guidelines

### Tier A: Executive
- Keep it short (1-2 pages max)
- Focus on "what" and "why"
- Minimal technical detail
- Clear value proposition

### Tier B: Tutorials
- Step-by-step format
- Copy-paste ready code
- Expected outcomes
- Common pitfalls

### Tier C: Reference
- Complete and precise
- Search-friendly structure
- All options documented
- Examples for each major use case

### Tier D: Technical
- Detailed explanations
- Mathematical notation OK
- Design rationale
- Research context

## Maintenance

### When to Update
- New features: Add to appropriate tier
- API changes: Update reference immediately
- Breaking changes: Update migration notes
- Bug fixes: Document in technical/fixes

### Review Checklist
- [ ] Active voice throughout
- [ ] Present tense
- [ ] Working code examples
- [ ] "See Also" section included
- [ ] Links verified
- [ ] Consistent formatting
- [ ] No broken references

---

**See Also:**
- [Documentation Architecture](ARCHITECTURE.md)
- [Migration Plan](MIGRATION_PLAN.md)
- [Documentation Index](INDEX.md)

