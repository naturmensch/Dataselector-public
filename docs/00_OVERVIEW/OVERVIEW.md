# Dataselector Documentation Hub

This hub keeps the **active thesis-v2 story** in front and demotes
secondary/historical material to clearly marked sections.

## Canonical thesis freeze

Start here if you want the current authoritative workflow:

- 🎓 [Thesis Pipeline How-To](../03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
- 📜 [Thesis Method Contract](../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md)
- 🧭 [Methodology](../08_GOVERNANCE/METHODOLOGY.md)
- ⚙️ [Config Policy](../08_GOVERNANCE/CONFIG_POLICY.md)
- 📚 [Parameter Policy Ledger](../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md)
- 📏 [Min-Distance Evidence Addendum](../06_REFERENCE/MIN_DISTANCE_EVIDENCE_ADDENDUM.md)
- 📐 [N-Samples Evidence Addendum](../06_REFERENCE/N_SAMPLES_EVIDENCE_ADDENDUM.md)

Canonical command:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

Direct validated-snapshot path:

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml
```

## Quick paths by audience

### Thesis / release users

- [03_USER_GUIDES/PIPELINES.md](../03_USER_GUIDES/PIPELINES.md)
- [03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md](../03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
- [THESIS_METHOD_CONTRACT.md](../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md)

### Developers and maintainers

- [DEVELOPER.md](../04_DEVELOPER/DEVELOPER.md)
- [02_THEORY/architecture.md](../02_THEORY/architecture.md)
- [TEST_SUITE_CURATION.md](../08_GOVERNANCE/TEST_SUITE_CURATION.md)
- [REPO_SURFACE_CURATION.md](../08_GOVERNANCE/REPO_SURFACE_CURATION.md)
- [04_DEVELOPER/](../04_DEVELOPER/)

### Secondary active reference

Useful, but not the default thesis entry story:

- [06_REFERENCE/api_reference.md](../06_REFERENCE/api_reference.md)
- [06_REFERENCE/TOOLS_REFERENCE.md](../06_REFERENCE/TOOLS_REFERENCE.md)
- [06_REFERENCE/thesis_decision_evidence/](../06_REFERENCE/thesis_decision_evidence/)
  for tracked parameter-decision evidence that remains referenced by active
  policy/config contracts
- [06_REFERENCE/scripts_reference.md](../06_REFERENCE/scripts_reference.md) for
  secondary / historical script context
- [05_ADVANCED/wandb_integration.md](../05_ADVANCED/wandb_integration.md)
- [05_ADVANCED/](../05_ADVANCED/)

### Operations and advanced / legacy context

- `generate-monitor` for run-local summaries of canonical thesis runs
- [07_ARCHIVE/legacy_xxl_ops/](../07_ARCHIVE/legacy_xxl_ops/) for archived XXL,
  monitor, systemd, and resume-era documentation

## Surface map

| Surface | Role |
|---|---|
| `README.md`, `README_EN.md`, this hub, thesis how-to, policy docs | Canonical active entry surface |
| API/tools reference, advanced docs, optional helper scripts | Secondary active surface |
| `docs/07_ARCHIVE/`, `tests/archive/`, legacy XXL and closeout docs | Historical / archived surface |

If you are unsure which path to use, prefer the canonical thesis freeze
workflow and then consult [REPO_SURFACE_CURATION.md](../08_GOVERNANCE/REPO_SURFACE_CURATION.md).

## Validation and release checks

Fast local governance:

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py
```

Long manual release checks:

```bash
micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py
make format-check
make test
```

## Evidence, historical, and archive locations

- [06_REFERENCE/thesis_decision_evidence/](../06_REFERENCE/thesis_decision_evidence/)
  for active thesis decision evidence with a stable repo-side home
- [07_ARCHIVE/](../07_ARCHIVE/) for repo-facing historical documentation
- `docs/reports/` for preserved historical or generated reports, not current
  evidence authority
- `tests/archive/` for historical tests
- `archive/` for retained repo history material
- `archive_local/` for local migration/backup mass that is not authoritative

## Reader guidance

1. If you want to run the thesis workflow, stop after the canonical section and
   use `thesis-orchestrate`.
 2. If you want to inspect live package boundaries, continue with
   [TEST_SUITE_CURATION.md](../08_GOVERNANCE/TEST_SUITE_CURATION.md) and
   [REPO_SURFACE_CURATION.md](../08_GOVERNANCE/REPO_SURFACE_CURATION.md).
3. If you are looking at scripts, XXL, monitors, or old reports, assume
   secondary or historical status unless an active doc says otherwise.
