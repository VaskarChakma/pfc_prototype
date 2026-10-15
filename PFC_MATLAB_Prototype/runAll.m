function results = runAll(mode)
%RUNALL Run the whole partner-free certification prototype and collect the results.
%   runAll("quick")  checks that everything runs, with tiny data and few steps (about 10 to 20 min).
%                    Its numbers are NOT meaningful.
%   runAll("full")   reproduces the full study. Expect several hours on one GPU.
%
%   Finished stages are saved and skipped on a re-run, so an interrupted run can be resumed.
%   Outputs in pfc_results_<mode>/ :
%     run_log.txt          full console log
%     pfc_results.json     all headline numbers in one file (send this file and run_log.txt back)
%     stress_nw.json, stress_ue.json, stress_summary.json, pfc_channels.mat (channel cache)

arguments
    mode (1,1) string {mustBeMember(mode, ["quick", "full"])} = "quick"
end
cfg = pfcConfig(mode);
if ~isfolder(cfg.OutDir), mkdir(cfg.OutDir); end
diary(fullfile(cfg.OutDir, "run_log.txt"));
cleanup = onCleanup(@() diary("off"));
fprintf("Partner-free certification prototype, mode %s, started %s\n", mode, string(datetime("now")));
fprintf("MATLAB %s\n", version);
v = ver; fprintf("  %s %s\n", [string({v.Name}); string({v.Version})]);
fprintf("GPU available: %d\n", canUseGPU);

results.mode = mode;
results.started = string(datetime("now"));

% 1) Requirement tests of the certification rule
tr = runtests("tPfcCertify");
results.unitTests = struct(passed=sum([tr.Passed]), failed=sum([tr.Failed]), total=numel(tr));
fprintf("unit tests: %d passed, %d failed\n", results.unitTests.passed, results.unitTests.failed);

% 2) Channels (cached)
t = tic; pfcMakeData(cfg); fprintf("channels ready in %.0f s\n", toc(t));

% 3) Stress study, both sides
for side = ["nw" "ue"]
    f = fullfile(cfg.OutDir, "stress_" + side + ".mat");
    if ~isfile(f)
        runPFCStress(side, cfg);
    end
    S = load(f, "out");
    outs.(side) = S.out;
end
summary = pfcAnalyzeStress(outs.nw, outs.ue, cfg.OutDir);
results.stress = rmfield(summary, "rows");

% 4) Theorem 1 Monte Carlo check
f = fullfile(cfg.OutDir, "theorem_check.mat");
if ~isfile(f)
    th = pfcTheoremCheck(cfg); save(f, "th");
end
S = load(f, "th"); results.theorem = S.th;

% 5) Accuracy against the eType II-like baseline
f = fullfile(cfg.OutDir, "accuracy_probe.mat");
if ~isfile(f)
    pr = pfcAccuracyProbe(cfg); save(f, "pr");
end
S = load(f, "pr"); results.accuracy = S.pr;

results.finished = string(datetime("now"));
writelines(jsonencode(results, PrettyPrint=true), fullfile(cfg.OutDir, "pfc_results.json"));
fprintf("\nAll done. Please send back:\n  %s\n  %s\n", fullfile(cfg.OutDir, "pfc_results.json"), ...
        fullfile(cfg.OutDir, "run_log.txt"));
end
