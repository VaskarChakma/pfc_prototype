function cfg = pfcConfig(mode)
%PFCCONFIG Settings for the partner-free certification prototype.
%   cfg = pfcConfig("full")  reproduces the full study (62 updates, 14 hidden partners per side).
%   cfg = pfcConfig("quick") runs the same code end to end with tiny data and few training
%                            steps. Use it first to check that everything runs (about 10 to 20 min).
%
%   All results are written to cfg.OutDir.

arguments
    mode (1,1) string {mustBeMember(mode, ["full", "quick"])} = "full"
end

cfg.Mode = mode;
cfg.OutDir = fullfile(pwd, "pfc_results_" + mode);

% Link and model
cfg.LatentDim = 64;          % 64 latents x 2 bits = 128-bit payload
cfg.K = 15;                  % certifiers per side
cfg.M = 8;                   % standard hidden partners per side
cfg.Delta = 0.01;            % tolerated SGCS loss
cfg.Alpha = 0.05;            % channel-sampling risk
cfg.BatchSize = 256;
cfg.UseGPU = true;           % used only if canUseGPU returns true

% Data sizes (channel drops)
if mode == "full"
    cfg.NTrain = 60000;      % per scenario (S0 and S1)
    cfg.NTest0 = 12000;      % certificate channels, scenario S0
    cfg.NTest1 = 4000;       % drift channels, scenario S1
    cfg.NTruth = 12000;      % independent channels for hidden-partner truth
    cfg.StepScale = 1;       % multiplies every training-step count
    cfg.TheoremTrials = 1500;
    cfg.TightnessTrials = 20000;
    cfg.ProbeSteps = 20000;
else
    cfg.NTrain = 3000;
    cfg.NTest0 = 1500;
    cfg.NTest1 = 500;
    cfg.NTruth = 1500;
    cfg.StepScale = 0.02;
    cfg.TheoremTrials = 60;
    cfg.TightnessTrials = 600;
    cfg.ProbeSteps = 200;
end
cfg.Steps = 1500;            % nominal steps for one partner or certifier
cfg.PairSteps = 4000;        % nominal steps for the reference encoder-decoder pair
end
