function models = pfcTrain(mode, models, data, opts)
%PFCTRAIN Custom training loop (Adam with cosine learning-rate decay).
%   mode "pair"   : models = {enc, dec}; data = {X, Re, Im}. End-to-end SGCS through the 2-bit quantizer.
%   mode "dec"    : models = {dec};      data = {Z, Re, Im}. Decoder trained on exchanged latents.
%   mode "encreg" : models = {enc};      data = {X, Zt}.     Encoder regresses the exchanged latents.
%   mode "proxy"  : models = {enc, frozenSurrogateDec};
%                   data = {Xnew, ReNew, ImNew, Xold, ReOld, ImOld[, Zold]}.
%                   Encoder update through a frozen surrogate decoder, with an old-data term
%                   (OldWeight) and, if Zold is given, a latent-anchor term (AnchorWeight).
%   All data arrays hold samples along their last dimension and must have the same count.
%   opts: Steps, LearnRate, BatchSize, Loss ("sgcs" | "nmse" | "l2" | "l1"), OldWeight,
%         AnchorWeight, Seed, UseGPU.

arguments
    mode (1,1) string {mustBeMember(mode, ["pair", "dec", "encreg", "proxy"])}
    models cell
    data cell
    opts.Steps (1,1) double {mustBePositive, mustBeInteger} = 1500
    opts.LearnRate (1,1) double = 1e-3
    opts.BatchSize (1,1) double = 256
    opts.Loss (1,1) string = "sgcs"
    opts.OldWeight (1,1) double = 1
    opts.AnchorWeight (1,1) double = 0
    opts.Seed (1,1) double = 0
    opts.UseGPU (1,1) logical = true
end
rng(opts.Seed, "twister");
useGPU = opts.UseGPU && canUseGPU;
if mode == "proxy", nTrain = 1; else, nTrain = numel(models); end   % proxy: only the encoder learns
avgG = cell(1, nTrain); avgSq = cell(1, nTrain);
N = size(data{1}, ndims(data{1}));
for it = 1:opts.Steps
    lr = 0.5 * opts.LearnRate * (1 + cos(pi * (it - 1) / opts.Steps));
    idx = randi(N, 1, opts.BatchSize);
    b = cellfun(@(d) pick(d, idx, useGPU), data, UniformOutput=false);
    nets = cellfun(@(m) m.net, models(1:nTrain), UniformOutput=false);
    % Each network is passed to dlfeval as its own argument so that it is traced.
    [~, grads] = dlfeval(@(varargin) lossFcn(varargin, models, b, mode, opts), nets{:});
    for k = 1:nTrain
        [models{k}.net, avgG{k}, avgSq{k}] = adamupdate(models{k}.net, grads{k}, avgG{k}, avgSq{k}, ...
                                                         it, lr, 0.9, 0.999, 1e-8);
    end
end
end

function d = pick(d, idx, useGPU)
if ndims(d) == 3, d = d(:, :, idx); else, d = d(:, idx); end
if useGPU, d = gpuArray(d); end
end

function [loss, grads] = lossFcn(nets, models, b, mode, opts)
for k = 1:numel(nets), models{k}.net = nets{k}; end
switch mode
    case "pair"
        Yad = pfcForward(models{2}, pfcQuant(pfcForward(models{1}, b{1})));
        loss = recon(Yad, b{2}, b{3}, "sgcs");
    case "dec"
        loss = recon(pfcForward(models{1}, b{1}), b{2}, b{3}, opts.Loss);
    case "encreg"
        e = tanh(pfcForward(models{1}, b{1})) - b{2};
        if opts.Loss == "l1", loss = mean(abs(e), "all"); else, loss = mean(e.^2, "all"); end
    case "proxy"
        sur = models{2};
        lNew = recon(pfcForward(sur, pfcQuant(pfcForward(models{1}, b{1}))), b{2}, b{3}, "sgcs");
        loss = lNew;
        if opts.OldWeight > 0
            lOld = recon(pfcForward(sur, pfcQuant(pfcForward(models{1}, b{4}))), b{5}, b{6}, "sgcs");
            loss = loss + opts.OldWeight * lOld;
        end
        if numel(b) >= 7 && opts.AnchorWeight > 0
            loss = loss + opts.AnchorWeight * mean((tanh(pfcForward(models{1}, b{4})) - b{7}).^2, "all");
        end
end
learn = cellfun(@(n) n.Learnables, nets, UniformOutput=false);
grads = cell(1, numel(nets));
[grads{:}] = dlgradient(loss, learn{:});
end

function L = recon(Yad, Re, Im, kind)
if kind == "nmse"      % squared error on unit-norm precoders, scaled by the antenna count
    [Hr, Hi] = pfcPrecoder(Yad);
    L = mean((Hr - Re).^2 + (Hi - Im).^2, "all") * size(Re, 1);
else
    L = 1 - mean(pfcSGCS(Yad, Re, Im));
end
end
