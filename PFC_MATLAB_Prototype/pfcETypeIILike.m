function [s, numCoef, bitsUsed] = pfcETypeIILike(Re, Im, payloadBits)
%PFCETYPEIILIKE Simplified eType II-like baseline at a given payload (base MATLAB).
%   [s, numCoef, bitsUsed] = pfcETypeIILike(Re, Im, 128) keeps the strongest angle-delay
%   coefficients, quantizes amplitude (3 bits, 3 dB steps relative to the strongest) and phase
%   (4 bits, relative to the strongest), and pays log2(nchoosek(416, L)) bits for their positions.
%   L is the largest count that fits the payload. Returns per-sample SGCS (1-by-N).
%   This is a simplified codebook, not the 3GPP eType II codebook. For the standard codebook use
%   nrCSIReportCSIRS with CodebookType "eType2" (5G Toolbox).
%   Re, Im: 32-by-13-by-N parts of unit-norm subband precoders.

arguments
    Re (:,:,:) single
    Im (:,:,:) single
    payloadBits (1,1) double = 128
end
nTx = 32; nSub = 13; nAll = nTx * nSub; ampBits = 3; phBits = 4;
log2nck = @(n, k) (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / log(2);
numCoef = 1;
while log2nck(nAll, numCoef + 1) + (numCoef + 1) * (ampBits + phBits) <= payloadBits
    numCoef = numCoef + 1;
end
bitsUsed = log2nck(nAll, numCoef) + numCoef * (ampBits + phBits);

W = double(permute(complex(Re, Im), [2 1 3]));          % subband x antenna x N
FF = fft(eye(nSub)) / sqrt(nSub); FS = fft(eye(nTx)) / sqrt(nTx);
N = size(W, 3); s = zeros(1, N);
lv = 10.^(-(0:2^ampBits - 1) * 3 / 20);
step = 2 * pi / 2^phBits;
for n = 1:N
    C = FF' * W(:, :, n) * conj(FS);
    c = C(:);
    [~, idx] = sort(abs(c), "descend");
    idx = idx(1:numCoef);
    sel = c(idx);
    amp = abs(sel) / (abs(sel(1)) + 1e-12);
    [~, a] = min(abs(amp - lv), [], 2);
    ph = angle(sel * exp(-1i * angle(sel(1))));
    cq = zeros(nAll, 1);
    cq(idx) = lv(a).' .* exp(1i * round(ph / step) * step);
    Wh = FF * reshape(cq, nSub, nTx) * FS.';
    num = abs(sum(conj(W(:, :, n)) .* Wh, 2)).^2;
    den = sum(abs(Wh).^2, 2) .* sum(abs(W(:, :, n)).^2, 2) + 1e-12;
    s(n) = mean(num ./ den);
end
end
