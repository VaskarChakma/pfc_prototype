function [Hr, Hi] = pfcPrecoder(Yad)
%PFCPRECODER Decoder output (angle-delay layout) to unit-norm precoders (dlarray-safe).
%   [Hr, Hi] = pfcPrecoder(Yad): Yad is 832-by-B in the pfcPrep layout. Hr and Hi are
%   32-by-13-by-B (antenna x subband x sample) real and imaginary parts of
%   W.' = FS * C.' * FF.', normalized per subband. Uses only reshape, permute and mtimes.

nSub = 13; nTx = 32;
Y = stripdims(Yad);
B = size(Y, 2);
T = reshape(Y, 2 * nTx, nSub * B);
Cr = T(1:nTx, :); Ci = T(nTx+1:end, :);                % C.' : beam x (delay, sample)
FS = fft(eye(nTx)) / sqrt(nTx); FF = fft(eye(nSub)) / sqrt(nSub);
if isa(Y, "dlarray"), proto = extractdata(Y(1)); else, proto = Y(1); end
like = @(A) cast(A, "like", proto);                     % same class and device (CPU or GPU) as the data
Ar = like(real(FS)); Ai = like(imag(FS));
Br = like(real(FF)); Bi = like(imag(FF));               % FF is symmetric, so FF.' = FF
% Step 1: antenna domain, T1 = FS * C.'
Tr = Ar * Cr - Ai * Ci;                                 % 32 x (13*B)
Ti = Ar * Ci + Ai * Cr;
% Step 2: frequency domain, H = T1 * FF.' computed as (FF * T1.').' per sample
Pr = reshape(permute(reshape(Tr, nTx, nSub, B), [2 1 3]), nSub, nTx * B);
Pi = reshape(permute(reshape(Ti, nTx, nSub, B), [2 1 3]), nSub, nTx * B);
Qr = Br * Pr - Bi * Pi;
Qi = Br * Pi + Bi * Pr;
Hr = permute(reshape(Qr, nSub, nTx, B), [2 1 3]);
Hi = permute(reshape(Qi, nSub, nTx, B), [2 1 3]);
nrm = sqrt(sum(Hr.^2 + Hi.^2, 1)) + 1e-9;
Hr = Hr ./ nrm; Hi = Hi ./ nrm;
end
