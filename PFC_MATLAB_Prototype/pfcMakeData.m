function data = pfcMakeData(cfg)
%PFCMAKEDATA Generate (or load from cache) all channel sets used by the study.
%   data.tr0  training channels, scenario S0 (exchanged dataset D is built from these)
%   data.tr1  training channels, scenario S1 (drifted conditions, new samples for updates)
%   data.te0  certificate channels, scenario S0 (used by the vendor to certify)
%   data.te1  drift channels, scenario S1 (reports SGCS after drift)
%   data.tp0  independent channels, scenario S0 (used only to measure hidden-partner truth)
%   Each set has fields X (832-by-N features), Re and Im (32-by-13-by-N precoder parts).

if ~isfolder(cfg.OutDir), mkdir(cfg.OutDir); end
cacheFile = fullfile(cfg.OutDir, "pfc_channels.mat");
if isfile(cacheFile)
    S = load(cacheFile, "data");
    data = S.data;
    return
end
sets = {"tr0", cfg.NTrain, "S0", 100
        "tr1", cfg.NTrain, "S1", 101
        "te0", cfg.NTest0, "S0", 102
        "te1", cfg.NTest1, "S1", 103
        "tp0", cfg.NTruth, "S0", 104};
for s = 1:size(sets, 1)
    t = tic;
    W = pfcCDLDataset(sets{s, 2}, sets{s, 3}, sets{s, 4});
    [X, W] = pfcPrep(W);
    Wt = permute(W, [2 1 3]);                      % antenna x subband x drop
    data.(sets{s, 1}) = struct(X=X, Re=single(real(Wt)), Im=single(imag(Wt)));
    fprintf("channels %s: %d drops of %s in %.0f s\n", sets{s, 1}, sets{s, 2}, sets{s, 3}, toc(t));
end
save(cacheFile, "data", "-v7.3");
end
