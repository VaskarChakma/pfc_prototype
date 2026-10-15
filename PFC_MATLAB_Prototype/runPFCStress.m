function out = runPFCStress(side, cfg)
%RUNPFCSTRESS Partner-free certification study for one side of the link.
%   out = runPFCStress("nw", cfg)  network vendor updates its decoder. Hidden partners are UE
%                                  encoders (TR 38.843 sub-option 4-1, NW-first dataset exchange).
%   out = runPFCStress("ue", cfg)  UE vendor updates its encoder. Hidden partners are gNB decoders.
%
%   Steps
%   1) The updating vendor trains its reference encoder and decoder on scenario S0 and shares
%      the dataset D = {target CSI, latent before quantization}.
%   2) 8 standard hidden partners and K = 15 certifiers are trained on D. They are drawn from one
%      pool of designs (MLP, CNN or attention; SGCS or NMSE loss; float or quantized latents;
%      all or half of D), so they are exchangeable by construction.
%   3) 6 stress partners are trained on D with designs outside the pool (larger or smaller models,
%      longer or shorter training, a quarter of D, or noisy inputs).
%   4) 30 (network side) or 32 (UE side) candidate updates are built, many of them close to the
%      tolerance.
%   5) Each update is certified with pfcCertify on the certificate channels. Hidden-partner truth
%      is measured on independent channels that the vendor never uses.
%   Results are saved as stress_<side>.mat and stress_<side>.json in cfg.OutDir.

arguments
    side (1,1) string {mustBeMember(side, ["nw", "ue"])}
    cfg struct = pfcConfig("full")
end
t0 = tic;
say = @(varargin) fprintf("[%6.0f s] %s\n", toc(t0), sprintf(varargin{:}));
data = pfcMakeData(cfg);
D = cfg.LatentDim; K = cfg.K; M = cfg.M; gpu = cfg.UseGPU;
st = @(n) max(5, round(n * cfg.StepScale));            % scaled step count
tr0 = data.tr0; tr1 = data.tr1;
N = size(tr0.X, 2);
Xmix = [tr0.X, tr1.X]; Rmix = cat(3, tr0.Re, tr1.Re); Imix = cat(3, tr0.Im, tr1.Im);
say("data ready: %d training drops per scenario", N);

%% 1) Reference pair and exchanged dataset
enc = pfcBuildModel("mlp", "enc", D, Width=[512 512], Seed=1);
dec = pfcBuildModel("mlp", "dec", D, Width=[512 512], Seed=2);
m = pfcTrain("pair", {enc, dec}, {tr0.X, tr0.Re, tr0.Im}, Steps=st(cfg.PairSteps), Seed=1, UseGPU=gpu);
enc = m{1}; dec = m{2};
tFloat = pfcEncode(enc, tr0.X, Output="float", UseGPU=gpu);       % shared latent before quantization
tQuant = quantVal(tFloat);
own = mean(pfcDecodeSGCS(dec, pfcEncode(enc, data.te0.X, UseGPU=gpu), data.te0.Re, data.te0.Im, UseGPU=gpu));
say("reference pair SGCS on S0 = %.4f", own);

%% 2) Implementation pool: standard partners and certifiers
arch.mlp = {struct(w=[512 512]), struct(w=1024), struct(w=[384 384 384])};
arch.cnn = {struct(c=12, h=256), struct(c=16, h=192)};
arch.attn = {struct(dm=64, blocks=2, heads=4), struct(dm=48, blocks=1, heads=4)};
rng(1000 + (side == "ue"), "twister");
kinds = repmat(["mlp" "cnn" "attn"], 1, 8); kinds = kinds(1:M + K); kinds = kinds(randperm(M + K));
pool = struct([]);
for i = 1:M + K
    a = arch.(kinds(i));
    c = struct(kind=kinds(i), cfg=a{randi(numel(a))}, loss=pickOne(["sgcs" "nmse"]), ...
               lat=pickOne(["float" "quant"]), frac=pickOne([1 0.5]), seed=i - 1, steps=cfg.Steps, noise=0);
    pool = [pool, c]; %#ok<AGROW>
end
order = randperm(M + K);
specs = [pool(order(1:M)), pool(order(M + 1:end))];
roles = [zeros(1, M), ones(1, K)];

%% 3) Stress partners (outside the pool design)
S = cfg.Steps;
stress = [spec("attn", struct(dm=96, blocks=3, heads=4), "sgcs", "float", 1.0, 50, 3 * S, 0)
          spec("cnn", struct(c=24, h=384), "nmse", "quant", 1.0, 51, 3 * S, 0)
          spec("mlp", struct(w=2048), "sgcs", "float", 0.25, 52, S, 0)
          spec("mlp", struct(w=[256 256]), "nmse", "float", 1.0, 53, S, 0.15)
          spec("cnn", struct(c=8, h=128), "sgcs", "quant", 0.25, 54, round(S / 2), 0)
          spec("attn", struct(dm=32, blocks=1, heads=2), "nmse", "float", 1.0, 55, S, 0.15)].';
specs = [specs, stress];
roles = [roles, 2 * ones(1, numel(stress))];

impl = cell(1, numel(specs));
roleName = ["partner" "certifier" "stress"];
for i = 1:numel(specs)
    impl{i} = trainImpl(specs(i), roles(i));
    say("trained %-9s %-4s loss=%s latent=%s share=%.2f", roleName(roles(i) + 1), specs(i).kind, ...
        specs(i).loss, specs(i).lat, specs(i).frac);
end

%% 4) Candidate updates
names = strings(0); cands = {};
if side == "nw"
    tf1 = pfcEncode(enc, tr1.X, UseGPU=gpu);           % new S1 samples labelled by the reference encoder
    zall = [tQuant, tf1];
    for s = [300 1500]
        r = pfcTrain("dec", {dec}, {zall, Rmix, Imix}, Steps=st(s), LearnRate=5e-4, Seed=s, UseGPU=gpu);
        addCand("ft_added_samples_" + s, r{1});
        r = pfcTrain("dec", {dec}, {tf1, tr1.Re, tr1.Im}, Steps=st(s), LearnRate=5e-4, Seed=s + 7, UseGPU=gpu);
        addCand("ft_new_only_" + s, r{1});
    end
    r = pfcTrain("dec", {pfcBuildModel("mlp", "dec", D, Width=[768 384], Seed=77)}, {zall, Rmix, Imix}, ...
                 Steps=st(3000), Seed=77, UseGPU=gpu);
    addCand("retrain_mlp_added_samples", r{1});
    r = pfcTrain("dec", {pfcBuildModel("cnn", "dec", D, Channels=12, Hidden=256, Seed=77)}, {zall, Rmix, Imix}, ...
                 Steps=st(3000), Seed=77, UseGPU=gpu);
    addCand("retrain_cnn_added_samples", r{1});
    r = pfcTrain("pair", {enc, dec}, {Xmix, Rmix, Imix}, Steps=st(1500), LearnRate=5e-4, Seed=8, UseGPU=gpu);
    addCand("joint_retrain_ref_pair", r{2});
    base = dec;
else
    sur = pfcTrain("dec", {pfcBuildModel("mlp", "dec", D, Width=[640 640], Seed=55)}, {tQuant, tr0.Re, tr0.Im}, ...
                   Steps=st(cfg.Steps), Seed=55, UseGPU=gpu);
    sur = sur{1};                                       % training surrogate, never a certifier
    for s = [50 300 1500]
        r = pfcTrain("pair", {enc, dec}, {Xmix, Rmix, Imix}, Steps=st(s), LearnRate=5e-4, Seed=s, UseGPU=gpu);
        addCand("ft_mix_" + s, r{1});
    end
    for s = [100 1000]
        r = pfcTrain("pair", {enc, dec}, {tr1.X, tr1.Re, tr1.Im}, Steps=st(s), LearnRate=5e-4, Seed=s + 3, UseGPU=gpu);
        addCand("ft_S1_only_" + s, r{1});
    end
    for wa = [0 0; 1 0; 1 10].'
        r = pfcTrain("proxy", {enc, sur}, {tr1.X, tr1.Re, tr1.Im, tr0.X, tr0.Re, tr0.Im, tFloat}, ...
                     Steps=st(1500), LearnRate=5e-4, OldWeight=wa(1), AnchorWeight=wa(2), Seed=9, UseGPU=gpu);
        addCand(sprintf("proxy_ft_old%g_anchor%g", wa(1), wa(2)), r{1});
    end
    r = pfcTrain("pair", {pfcBuildModel("cnn", "enc", D, Channels=12, Hidden=256, Seed=88), ...
                          pfcBuildModel("mlp", "dec", D, Width=[512 512], Seed=89)}, {Xmix, Rmix, Imix}, ...
                 Steps=st(3000), Seed=88, UseGPU=gpu);
    addCand("retrain_cnn_from_scratch", r{1});
    base = enc;
end
for s = [0.05 0.15 0.3]
    addCand("weight_noise_" + s, pfcModifyWeights(base, "noise", s, round(s * 100)));
end
for f = [0.3 0.6 0.85]
    addCand("prune_" + round(f * 100) + "pct", pfcModifyWeights(base, "prune", f));
end
for b = [8 6 4]
    addCand("fixed_point_int" + b, pfcModifyWeights(base, "fixed", b));
end
for s = [0.06 0.08 0.10 0.12]                           % updates close to the tolerance
    addCand("weight_noise_" + s, pfcModifyWeights(base, "noise", s, round(s * 1000)));
end
for f = [0.15 0.2 0.25 0.35 0.4 0.45]
    addCand("prune_" + round(f * 100) + "pct", pfcModifyWeights(base, "prune", f));
end
for b = [7 5]
    addCand("fixed_point_int" + b, pfcModifyWeights(base, "fixed", b));
end
if side == "nw"
    for s = [100 600]
        r = pfcTrain("dec", {dec}, {zall, Rmix, Imix}, Steps=st(s), LearnRate=5e-4, Seed=s + 11, UseGPU=gpu);
        addCand("ft_added_samples_" + s, r{1});
    end
else
    for s = [20 50]
        r = pfcTrain("pair", {enc, dec}, {tr1.X, tr1.Re, tr1.Im}, Steps=st(s), LearnRate=5e-4, Seed=s + 13, UseGPU=gpu);
        addCand("ft_S1_only_" + s, r{1});
    end
end
say("%d candidate updates", numel(names));

%% 5) Evaluation: certificate channels (te0), truth channels (tp0), drift channels (te1)
nI = numel(impl); nC = numel(cands);
allModels = [{base}, cands];                            % column 1 = old model
S0 = zeros(size(data.te0.X, 2), nI, nC + 1, "single");
Sb = zeros(size(data.tp0.X, 2), nI, nC + 1, "single");
S1 = zeros(nI, nC + 1);
sets = {data.te0, data.tp0, data.te1};
if side == "nw"                                         % implementations are encoders
    for i = 1:nI
        L = cellfun(@(d) pfcEncode(impl{i}, d.X, UseGPU=gpu), sets, UniformOutput=false);
        for c = 1:nC + 1
            S0(:, i, c) = pfcDecodeSGCS(allModels{c}, L{1}, sets{1}.Re, sets{1}.Im, UseGPU=gpu);
            Sb(:, i, c) = pfcDecodeSGCS(allModels{c}, L{2}, sets{2}.Re, sets{2}.Im, UseGPU=gpu);
            S1(i, c) = mean(pfcDecodeSGCS(allModels{c}, L{3}, sets{3}.Re, sets{3}.Im, UseGPU=gpu));
        end
        say("evaluated implementation %d of %d", i, nI);
    end
else                                                    % candidates are encoders
    for c = 1:nC + 1
        L = cellfun(@(d) pfcEncode(allModels{c}, d.X, UseGPU=gpu), sets, UniformOutput=false);
        for i = 1:nI
            S0(:, i, c) = pfcDecodeSGCS(impl{i}, L{1}, sets{1}.Re, sets{1}.Im, UseGPU=gpu);
            Sb(:, i, c) = pfcDecodeSGCS(impl{i}, L{2}, sets{2}.Re, sets{2}.Im, UseGPU=gpu);
            S1(i, c) = mean(pfcDecodeSGCS(impl{i}, L{3}, sets{3}.Re, sets{3}.Im, UseGPU=gpu));
        end
        say("evaluated candidate %d of %d", c - 1, nC);
    end
end

%% 6) Certification decisions and hidden-partner truth
isCert = roles == 1; isStd = roles == 0; isStress = roles == 2;
res = struct([]);
for c = 1:nC
    dS = double(S0(:, :, c + 1) - S0(:, :, 1));         % per-channel change, certificate channels
    truth = mean(double(Sb(:, :, c + 1) - Sb(:, :, 1)), 1);
    [ok, rep] = pfcCertify(dS(:, isCert), cfg.Delta, cfg.Alpha, 1, M);
    r = struct(name=names(c), accept=ok, certMinLcb=rep.decisionStatistic, ...
               worstStd=min(truth(isStd)), worstStress=min(truth(isStress)), ...
               brokenStd=sum(truth(isStd) < -cfg.Delta), brokenStress=sum(truth(isStress) < -cfg.Delta), ...
               truth=truth, certSetMean=mean(dS, 1), certLcb=rep.lcb, s1=S1(:, c + 1).', s1Old=S1(:, 1).');
    res = [res, r]; %#ok<AGROW>
    say("%-28s accept=%d certLCB %+.4f worst std %+.4f (%d/%d) worst stress %+.4f (%d/%d)", names(c), ok, ...
        r.certMinLcb, r.worstStd, r.brokenStd, M, r.worstStress, r.brokenStress, sum(isStress));
end
out = struct(side=side, mode=cfg.Mode, ownPairSGCS=own, roles=roles, kinds=[specs.kind], ...
             delta=cfg.Delta, alpha=cfg.Alpha, K=K, M=M, cands=res);
save(fullfile(cfg.OutDir, "stress_" + side + ".mat"), "out");
writelines(jsonencode(out, PrettyPrint=true), fullfile(cfg.OutDir, "stress_" + side + ".json"));
say("done: %s", fullfile(cfg.OutDir, "stress_" + side + ".json"));

%% nested helpers
    function addCand(name, mdl)
        names(end + 1) = string(name);
        cands{end + 1} = mdl;
    end

    function model = trainImpl(c, roleKey)
        rng(10000 + c.seed + 500 * roleKey, "twister");
        n = round(N * c.frac);
        if c.lat == "float", tgt = tFloat(:, 1:n); else, tgt = tQuant(:, 1:n); end
        steps = st(c.steps);
        if side == "ue"     % counterpart is a decoder trained on (latent -> target CSI)
            mdl = buildFamily(c, "dec");
            zin = tgt + single(c.noise) * randn(size(tgt), "single");
            r2 = pfcTrain("dec", {mdl}, {zin, tr0.Re(:, :, 1:n), tr0.Im(:, :, 1:n)}, Steps=steps, ...
                          Loss=c.loss, Seed=c.seed, UseGPU=gpu);
        else                % counterpart is a UE encoder trained to reproduce the shared latents
            mdl = buildFamily(c, "enc");
            xin = tr0.X(:, 1:n);
            xin = xin + single(c.noise) * std(xin(:)) * randn(size(xin), "single");
            lossName = "l2"; if c.loss == "nmse", lossName = "l1"; end
            r2 = pfcTrain("encreg", {mdl}, {xin, tgt}, Steps=steps, Loss=lossName, Seed=c.seed, UseGPU=gpu);
        end
        model = r2{1};
    end

    function mdl = buildFamily(c, role)
        sd = 10000 + c.seed;
        switch c.kind
            case "mlp",  mdl = pfcBuildModel("mlp", role, D, Width=c.cfg.w, Seed=sd);
            case "cnn",  mdl = pfcBuildModel("cnn", role, D, Channels=c.cfg.c, Hidden=c.cfg.h, Seed=sd);
            case "attn", mdl = pfcBuildModel("attn", role, D, Dim=c.cfg.dm, Blocks=c.cfg.blocks, ...
                                             Heads=c.cfg.heads, Seed=sd);
        end
    end
end

function v = pickOne(options)
v = options(randi(numel(options)));
end

function c = spec(kind, a, loss, lat, frac, seed, steps, noise)
c = struct(kind=kind, cfg=a, loss=loss, lat=lat, frac=frac, seed=seed, steps=steps, noise=noise);
end

function q = quantVal(t)
q = -1 + 0.5 * (min(max(floor((t + 1) / 0.5), 0), 3) + 0.5);
end
