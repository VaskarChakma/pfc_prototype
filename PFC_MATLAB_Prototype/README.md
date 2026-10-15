# Partner-free compatibility certification: MATLAB prototype

This folder holds the full prototype for the proposal "Partner-Free Compatibility Certification for Two-Sided AI CSI Feedback in Multi-Vendor 6G Networks". It covers the whole study, from CDL channel generation to the final summary table.

## Requirements
- MATLAB R2024b or later is recommended. R2023b is the minimum, because the attention layers need it.
- 5G Toolbox and Deep Learning Toolbox.
- Parallel Computing Toolbox is optional. With it, training runs on a supported NVIDIA GPU and channel generation runs on parallel workers. Without it, everything runs on the CPU, but much more slowly.

## How to run
1. Put all files in one folder and set it as the MATLAB current folder.
2. Run the quick check first:
   ```matlab
   runAll("quick")
   ```
   This uses tiny data and very short training, so it should finish in roughly 10 to 30 minutes. Its numbers mean nothing. It only shows that every stage runs on your MATLAB release. If it stops with an error, send me the error message and `pfc_results_quick/run_log.txt`.
3. Run the full study:
   ```matlab
   runAll("full")
   ```
   It trains about 80 networks and evaluates 62 updates against 29 models. Expect several hours on one GPU, or much longer on a CPU. Each finished stage is saved, so if the run stops you can call `runAll("full")` again to continue.
4. Send back these two files from `pfc_results_full/`:
   - `pfc_results.json`: all headline numbers
   - `run_log.txt`: the full console log
   If possible, also send `stress_nw.json`, `stress_ue.json` and `stress_summary.json`.

## What each file does
| File | Purpose |
|---|---|
| `runAll.m` | Runs everything: requirement tests, channels, both sides of the stress study, the summary, the Theorem 1 check and the accuracy check |
| `pfcConfig.m` | All settings for the "full" and "quick" modes |
| `pfcCDLDataset.m` | TR 38.901 CDL channels from `nrCDLChannel`. Each drop draws a new delay spread, array bearing and random seed. The subband precoder is the channel at the centre subcarrier of each 4-RB subband |
| `pfcMakeData.m` | Generates and caches the five channel sets (training S0 and S1, certificate S0, drift S1, independent truth S0) |
| `pfcPrep.m` | Phase normalization and angle-delay features |
| `pfcBuildModel.m` | MLP, CNN and attention encoders and decoders as `dlnetwork` objects |
| `pfcForward.m`, `pfcPrecoder.m`, `pfcSGCS.m`, `pfcQuant.m` | Forward pass, conversion to unit-norm precoders, SGCS, and the 2-bit quantizer with a straight-through gradient |
| `pfcTrain.m` | Custom training loop (Adam, cosine decay) for the reference pair, decoders, encoders and proxy updates |
| `pfcEncode.m`, `pfcDecodeSGCS.m` | Batched evaluation on CPU or GPU |
| `pfcModifyWeights.m` | Weight-noise, pruning and fixed-point updates |
| `pfcCertify.m` | The certification rule and the Theorem 1 risk bounds |
| `runPFCStress.m` | The stress study for one side ("nw" or "ue"): 8 standard partners, 15 certifiers, 6 stress partners and 30 or 32 updates |
| `pfcAnalyzeStress.m` | Summary over both sides: wrong keeps, cautious rejections, the baseline rules, the rank-event rate, drift gains and fixed-point results |
| `pfcTheoremCheck.m` | Monte Carlo check of Theorem 1 (base MATLAB only) |
| `pfcETypeIILike.m`, `pfcAccuracyProbe.m` | Simplified eType II-like baseline at 128 bits, and a larger autoencoder for comparison |
| `tPfcCertify.m` | Seven requirement tests for the rule. Run with `runtests("tPfcCertify")` |

## Notes
- On releases older than R2024b, `pfcCDLDataset` uses path gains with `nrPerfectChannelEstimate` instead of `ChannelResponseOutput="ofdm-response"`.
- To reduce run time, lower `NTrain`, `NTest0` or `NTruth` in `pfcConfig.m`. Smaller test sets make the confidence bounds wider, so the rule rejects more updates.
- `pfcETypeIILike` is a simplified codebook. For the standard eType II codebook, use `nrCSIReportCSIRS` with `CodebookType="eType2"`.
