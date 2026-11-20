# Final results (mean ± std over seeds)

## 1. Main table

| method | seeds | params (M) | SR PSNR | SR SSIM | chroma MAE | class-flip | label err (judge reads colour) | ms/tile no-TTA | ms/tile TTA-8 |
|---|---|---|---|---|---|---|---|---|---|
| **tta8** | | | | | | | | | |
| arc | 2 | 1.28 | 35.2979 ± 0.0569 | 0.9653 ± 0.0004 | 0.0325 ± 0.0000 | 0.2172 ± 0.0015 | 0.4580 ± 0.0113 | 32.5 | 254.1 |
| reg | 3 | 1.07 | - | - | 0.0326 ± 0.0002 | 0.2222 ± 0.0034 | 0.4487 ± 0.0113 | 6.6 | 56.3 |
| regvar | 1 | 1.07 | - | - | 0.0332 ± 0.0000 | 0.2220 ± 0.0000 | 0.4516 ± 0.0000 | 3.9 | 24.8 |
| pix2pix | 3 | 1.07 | - | - | 0.0328 ± 0.0005 | 0.2157 ± 0.0011 | 0.4456 ± 0.0027 | 6.5 | 48.5 |
| sr | 3 | 1.07 | 35.3531 ± 1.4127 | 0.9676 ± 0.0190 | - | - | - | 4.3 | 35.6 |
| **no_tta** | | | | | | | | | |
| arc | 2 | 1.28 | 35.2612 ± 0.0531 | 0.9656 ± 0.0005 | 0.0325 ± 0.0000 | 0.2175 ± 0.0013 | 0.4581 ± 0.0112 | 32.5 | 254.1 |
| reg | 3 | 1.07 | - | - | 0.0326 ± 0.0001 | 0.2224 ± 0.0034 | 0.4486 ± 0.0114 | 6.6 | 56.3 |
| regvar | 1 | 1.07 | - | - | 0.0333 ± 0.0000 | 0.2221 ± 0.0000 | 0.4516 ± 0.0000 | 3.9 | 24.8 |
| pix2pix | 3 | 1.07 | - | - | 0.0334 ± 0.0007 | 0.2185 ± 0.0013 | 0.4457 ± 0.0027 | 6.5 | 48.5 |
| sr | 3 | 1.07 | 35.2690 ± 1.3555 | 0.9674 ± 0.0188 | - | - | - | 4.3 | 35.6 |

## 2. Ablations

| variant (TTA-8) | chroma MAE | class-flip | label err |
|---|---|---|---|
| ARC-IR hybrid point head | 0.0325 ± 0.0000 | 0.2172 ± 0.0015 | 0.4580 ± 0.0113 |
| ARC-IR flow-sample mean (same model) | 0.0333 ± 0.0005 | 0.2301 ± 0.0003 | 0.4162 ± 0.0144 |

## 3. Selective risk (class-flip) — lower is better; first seed of each method

| method | score | AURC | risk at 50% coverage | 70% | 90% | calibrated commit: coverage / risk (alpha=0.15) |
|---|---|---|---|---|---|---|
| final_arc_s0 | seg_confidence | 0.1388 | 0.1181 | 0.1432 | 0.1917 | 1.00 / 0.2153 |
| final_arc_s0 | class_head_entropy | 0.2952 | 0.2761 | 0.2361 | 0.2187 | 1.00 / 0.2153 |
| final_arc_s0 | flow_spread | 0.1529 | 0.1431 | 0.1491 | 0.1881 | 1.00 / 0.2153 |
| final_arc_s0 | random | 0.2154 | 0.2153 | 0.2154 | 0.2153 | 1.00 / 0.2153 |
| final_regvar_s0 | seg_confidence | 0.1313 | 0.1222 | 0.1476 | 0.1931 | 1.00 / 0.2186 |
| final_regvar_s0 | learned_sigma | 0.1562 | 0.1366 | 0.1501 | 0.1897 | 1.00 / 0.2186 |
| final_regvar_s0 | random | 0.2187 | 0.2185 | 0.2186 | 0.2186 | 1.00 / 0.2186 |
| final_reg_s0 | seg_confidence | 0.1471 | 0.1284 | 0.1465 | 0.1903 | 1.00 / 0.2145 |
| final_reg_s0 | random | 0.2146 | 0.2145 | 0.2146 | 0.2145 | 1.00 / 0.2145 |
| final_pix2pix_s0 | seg_confidence | 0.1434 | 0.1288 | 0.1505 | 0.1934 | 1.00 / 0.212 |
| final_pix2pix_s0 | random | 0.2121 | 0.2120 | 0.2120 | 0.2120 | 1.00 / 0.212 |
| reg_ensemble | seg_confidence | 0.1328 | 0.1186 | 0.1429 | 0.1884 | 1.00 / 0.2131 |
| reg_ensemble | ensemble_spread | 0.1535 | 0.1452 | 0.1654 | 0.1974 | 1.00 / 0.2131 |
| reg_ensemble | random | 0.2132 | 0.2131 | 0.2132 | 0.2132 | 1.00 / 0.2131 |

## 4. Conformal colour intervals (delta=0.1; radius scale from calibration regions)

| method | interval | calibration | test coverage | worst-region coverage | coverage std over regions | mean radius | constant radius at same coverage |
|---|---|---|---|---|---|---|---|
| final_arc_s0 | adaptive | pooled | 0.892 | 0.651 | 0.119 | 0.0600 | 0.0664 |
| final_arc_s0 | adaptive | region_quantile | 0.912 | 0.705 | 0.101 | 0.0640 | 0.0729 |
| final_arc_s0 | constant | pooled | 0.880 | 0.564 | 0.151 | 0.0624 | 0.0624 |
| final_arc_s0 | constant | region_quantile | 0.953 | 0.795 | 0.074 | 0.0969 | 0.0969 |
| final_regvar_s0 | adaptive | pooled | 0.891 | 0.627 | 0.127 | 0.0748 | 0.0646 |
| final_regvar_s0 | adaptive | region_quantile | 0.907 | 0.664 | 0.116 | 0.0796 | 0.0706 |
| final_regvar_s0 | constant | pooled | 0.882 | 0.569 | 0.149 | 0.0615 | 0.0615 |
| final_regvar_s0 | constant | region_quantile | 0.938 | 0.731 | 0.097 | 0.0875 | 0.0875 |
| final_reg_s0 | constant | pooled | 0.888 | 0.590 | 0.144 | 0.0627 | 0.0627 |
| final_reg_s0 | constant | region_quantile | 0.944 | 0.758 | 0.088 | 0.0924 | 0.0924 |
| final_pix2pix_s0 | constant | pooled | 0.890 | 0.612 | 0.135 | 0.0625 | 0.0625 |
| final_pix2pix_s0 | constant | region_quantile | 0.956 | 0.803 | 0.071 | 0.1012 | 0.1012 |
| reg_ensemble | adaptive | pooled | 0.923 | 0.788 | 0.065 | 0.0803 | 0.0764 |
| reg_ensemble | adaptive | region_quantile | 0.956 | 0.864 | 0.044 | 0.0978 | 0.0993 |
| reg_ensemble | constant | pooled | 0.883 | 0.571 | 0.150 | 0.0603 | 0.0603 |
| reg_ensemble | constant | region_quantile | 0.943 | 0.752 | 0.089 | 0.0889 | 0.0889 |