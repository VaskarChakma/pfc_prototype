function [X, W] = pfcPrep(W)
%PFCPREP Phase-normalize precoders and build angle-delay features.
%   [X, W] = pfcPrep(W) takes W (13 x 32 x N complex, unit-norm per subband) and returns
%     X  (2*32*13)-by-N single features. For each subband k (slowest index) the 64 values are
%        [real(C(k,1:32)) imag(C(k,1:32))], where C = FF' * W * conj(FS) is the unitary
%        angle-delay transform, scaled by 3
%     W  the precoders rotated by one common phase so the strongest angle-delay coefficient
%        is real-positive (SGCS is unchanged)

[nSub, nTx, N] = size(W);
FF = fft(eye(nSub)) / sqrt(nSub);
FS = fft(eye(nTx)) / sqrt(nTx);
C = pagemtimes(pagemtimes(FF', W), conj(FS));
Cf = reshape(C, nSub * nTx, N);
[~, iMax] = max(abs(Cf), [], 1);
ph = exp(-1i * angle(Cf(sub2ind(size(Cf), iMax, 1:N))));
C = C .* reshape(ph, 1, 1, N);
W = W .* reshape(ph, 1, 1, N);
A = cat(2, real(C), imag(C));                                   % nSub x 64 x N
X = single(3 * reshape(permute(A, [2 1 3]), 2 * nTx * nSub, N)); % 64 fastest, then subband
end
