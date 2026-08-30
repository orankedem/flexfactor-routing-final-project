# Frozen configuration exports

`policy_final.json` is committed directly because the final reference policy is small.

The exact A1/A2/A3 XGBoost metadata is stored with the frozen Drive artifacts. During the final showcase, the notebook calls `export_exact_configs(...)` and creates:

- `a1_final.json`
- `a2_final.json`
- `a3_final.json`

Those compact files preserve exact model parameters, final tree count, feature list, stage architecture and temporal-history configuration while intentionally omitting the large categorical-level maps.

After the showcase notebook runs successfully, upload those three generated JSON files to this folder in GitHub and then save the executed notebook back to GitHub.
