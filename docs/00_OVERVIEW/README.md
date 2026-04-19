# Dataselector Documentation

Welcome to the complete Dataselector documentation hub. This repository is organized into clear, structured categories to help you find exactly what you need.

## Navigation

### Getting Started
- **[Quick Start](../01_QUICK_START/)** – First steps and setup for new users
- **[Overview & Index](OVERVIEW.md)** – Complete navigation guide

### Core Knowledge
- **[Theory & Concepts](../02_THEORY/)** – Scientific foundations, architecture, methodology
- **[User Guides](../03_USER_GUIDES/)** – Thesis pipeline, sampling, validation workflows
- **[Advanced Topics](../05_ADVANCED/)** – Optimization, tuning, W&B integration
- **[References](../06_REFERENCE/)** – API docs, tools, evidence, decision records

### Development & Maintenance
- **[Developer Guide](../04_DEVELOPER/)** – Setup, architecture, contribution guidelines
- **[Governance](../08_GOVERNANCE/)** – Policies, configuration, principles
- **[Internal & Planning](../09_INTERNAL/)** – ADRs, reports, architecture notes, status

### Archive
- **[07_ARCHIVE](../07_ARCHIVE/)** – Historical workflows, legacy docs, closeout reports

---

## Quick Reference

### Canonical Commands

**Thesis workflow (recommended):**
```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

**Check latest documentation:**
- Policies: [08_GOVERNANCE/CONFIG_POLICY.md](../08_GOVERNANCE/CONFIG_POLICY.md)
- Runtime: [04_DEVELOPER/DEVELOPER.md](../04_DEVELOPER/DEVELOPER.md)
- Methodology: [08_GOVERNANCE/METHODOLOGY.md](../08_GOVERNANCE/METHODOLOGY.md)

### Organization Principles

| Section | Purpose | Audience |
|---------|---------|----------|
| 00_OVERVIEW | Hub & navigation | Everyone |
| 01_QUICK_START | First-time setup | New users |
| 02_THEORY | Foundations & design | Data scientists, developers |
| 03_USER_GUIDES | Operational workflows | Researchers, end users |
| 04_DEVELOPER | Code & contribution | Contributors, maintainers |
| 05_ADVANCED | Deep-dive topics | Advanced users |
| 06_REFERENCE | API, tools, evidence | Researchers, developers |
| 07_ARCHIVE | Historical material | Archival reference |
| 08_GOVERNANCE | Policies & standards | Maintainers, leads |
| 09_INTERNAL | Planning &  reports | Internal team |

---

## Recent Changes

**Phase 2026-04-19:** Complete docs restructuring
- Consolidated all root markdown files into logical categories
- Created 08_GOVERNANCE for policies and contracts
- Created 09_INTERNAL for planning and reports
- Archived loose orderdirectories under 09_INTERNAL
- All content preserved, no deletions

---

## Finding Help

1. **Lost?** Start with [01_QUICK_START/](../01_QUICK_START/)
2. **Want full picture?** Read [02_THEORY/architecture.md](../02_THEORY/architecture.md)
3. **Need to run something?** Go to [03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md](../03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
4. **Contributing code?** Check [04_DEVELOPER/DEVELOPER.md](../04_DEVELOPER/DEVELOPER.md)
5. **Questions?** Review [08_GOVERNANCE/CONFIG_POLICY.md](../08_GOVERNANCE/CONFIG_POLICY.md) for policy decisions

---

**Last Updated:** 2026-04-19  
**Status:** Complete restructure ✓
