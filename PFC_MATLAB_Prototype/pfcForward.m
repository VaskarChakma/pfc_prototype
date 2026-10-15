function Y = pfcForward(model, X)
%PFCFORWARD Run an encoder or decoder on canonical-layout data (layout of pfcPrep).
%   Encoder: X is 832-by-B features; Y is D-by-B pre-tanh latent ("CB" dlarray).
%   Decoder: X is D-by-B latent;     Y is 832-by-B angle-delay output in the pfcPrep layout.

nSub = 13; nTx = 32;
X = stripdims(dlarray(X));
B = size(X, 2);
if model.role == "enc"
    T = reshape(X, 2 * nTx, nSub, B);                 % 64 x subband x B
    switch model.kind
        case "mlp"
            in = dlarray(X, "CB");
        case "cnn"   % image: nSub x nTx x 2 x B
            re = T(1:nTx, :, :); im = T(nTx+1:end, :, :);
            in = dlarray(cat(3, permute(re, [2 1 4 3]), permute(im, [2 1 4 3])), "SSCB");
        case "attn"  % one token per subband, 64 channels; stored order C x B x T
            in = dlarray(permute(T, [1 3 2]), "CBT");
    end
    Y = forward(model.net, in);
else
    Z = stripdims(forward(model.net, dlarray(X, "CB")));
    switch model.kind
        case "mlp"
            Y = Z;
        case "cnn"   % nSub x nTx x 2 x B -> 64 x nSub x B
            re = permute(Z(:, :, 1, :), [2 1 4 3]);
            im = permute(Z(:, :, 2, :), [2 1 4 3]);
            Y = reshape(cat(1, re, im), 2 * nTx * nSub, B);
        case "attn"  % 64 x B x nSub -> 64 x nSub x B
            Y = reshape(permute(Z, [1 3 2]), 2 * nTx * nSub, B);
    end
    Y = dlarray(Y, "CB");
end
end
