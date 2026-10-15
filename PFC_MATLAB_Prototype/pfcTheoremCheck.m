function res = pfcTheoremCheck(cfg)
%PFCTHEOREMCHECK Monte Carlo check of Theorem 1, independent of the CSI models (base MATLAB).
%   Part 1 (worst case over populations): K certifiers and one partner draw true SGCS changes
%   mu from one heavy-tailed law (Student t with 3 degrees of freedom, swept over centre and
%   spread). Certifiers see N noisy paired samples. Rule: accept iff the j-th smallest LCB >= -delta.
%   Reports the worst P(accept and partner mu < -delta) against the bound j/(K+1) + alpha.
%   Part 2 (tightness): exactly one fragile implementation among K+1 exchangeable ones.

arguments
    cfg struct = pfcConfig("full")
end
rng(0, "twister");
delta = 0.01; alpha = 0.05; N = 4000;
trials = cfg.TheoremTrials;
settings = [9 1; 15 1; 15 2; 19 1];
res.worstCase = zeros(size(settings, 1), 4);
fprintf("%3s %2s %30s %22s\n", "K", "j", "worst-case P(accept & broken)", "bound j/(K+1)+alpha");
for s = 1:size(settings, 1)
    K = settings(s, 1); j = settings(s, 2); worst = 0;
    for centre = linspace(-0.02, 0.02, 21)
        for spread = [0.001 0.003 0.01]
            hits = 0;
            for t = 1:trials
                mu = centre + spread * trnd3(K + 1);
                hits = hits + oneTrial(mu, K, j, N, delta, alpha);
            end
            worst = max(worst, hits / trials);
        end
    end
    res.worstCase(s, :) = [K, j, worst, j / (K + 1) + alpha];
    fprintf("%3d %2d %30.4f %22.4f\n", K, j, worst, j / (K + 1) + alpha);
end
res.tightness = zeros(2, 3);
for s = 1:2
    K = [9 15]; K = K(s); hits = 0;
    for t = 1:cfg.TightnessTrials
        mu = zeros(1, K + 1); mu(randi(K + 1)) = -0.05;
        hits = hits + oneTrial(mu, K, 1, N, delta, alpha);
    end
    res.tightness(s, :) = [K, hits / cfg.TightnessTrials, 1 / (K + 1)];
    fprintf("tightness K=%d: P(accept & broken) = %.4f vs rank term 1/(K+1) = %.4f\n", K, hits / cfg.TightnessTrials, 1 / (K + 1));
end
end

function hit = oneTrial(mu, K, j, N, delta, alpha)
x = max(min(mu(1:K) + 0.02 * randn(N, K), 1), -1);     % paired per-channel differences
ok = pfcCertify(x, delta, alpha, j);
hit = ok && mu(K + 1) < -delta;
end

function v = trnd3(n)
% Student t with 3 degrees of freedom from normals (no Statistics Toolbox needed)
z = randn(1, n); c = sum(randn(3, n).^2, 1);
v = z ./ sqrt(c / 3);
end
