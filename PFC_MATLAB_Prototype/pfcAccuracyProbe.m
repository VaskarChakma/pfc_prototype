function res = pfcAccuracyProbe(cfg)
%PFCACCURACYPROBE Can a larger learned two-sided code beat the eType II-like baseline at 128 bits?
%   Trains an MLP encoder and decoder with two hidden layers of 2048 units (64 latents x 2 bits)
%   on scenario S0 with batch size 512, and compares test SGCS with pfcETypeIILike at 128 bits.

arguments
    cfg struct = pfcConfig("full")
end
data = pfcMakeData(cfg);
te = data.te0;
[sb, L, bits] = pfcETypeIILike(te.Re, te.Im, 128);
res.eTypeIILike = mean(sb); res.eTypeIILikeCoefficients = L; res.eTypeIILikeBits = bits;
fprintf("eType II-like baseline: SGCS %.4f with %d coefficients, %.1f bits\n", res.eTypeIILike, L, bits);

enc = pfcBuildModel("mlp", "enc", cfg.LatentDim, Width=[2048 2048], Seed=1);
dec = pfcBuildModel("mlp", "dec", cfg.LatentDim, Width=[2048 2048], Seed=2);
m = pfcTrain("pair", {enc, dec}, {data.tr0.X, data.tr0.Re, data.tr0.Im}, Steps=cfg.ProbeSteps, ...
             BatchSize=512, Seed=0, UseGPU=cfg.UseGPU);
z = pfcEncode(m{1}, te.X, UseGPU=cfg.UseGPU);
res.mlp2048 = mean(pfcDecodeSGCS(m{2}, z, te.Re, te.Im, UseGPU=cfg.UseGPU));
fprintf("MLP 2048x2 autoencoder, 128 bits: SGCS %.4f\n", res.mlp2048);
end
