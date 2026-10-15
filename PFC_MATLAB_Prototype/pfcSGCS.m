function s = pfcSGCS(Yad, Wre, Wim)
%PFCSGCS Squared generalized cosine similarity per sample (dlarray-safe).
%   s = pfcSGCS(Yad, Wre, Wim): Yad is the 832-by-B decoder output (pfcPrep layout); Wre/Wim are
%   32-by-13-by-B real/imag parts of the unit-norm targets (antenna x subband x sample).
%   Returns 1-by-B SGCS averaged over subbands: mean_k |w_k^H w^_k|^2 (both unit norm).
[Hr, Hi] = pfcPrecoder(Yad);
B = size(Hr, 3);
num = sum(Wre .* Hr + Wim .* Hi, 1).^2 + sum(Wre .* Hi - Wim .* Hr, 1).^2;
s = reshape(mean(num, 2), 1, B);
end
