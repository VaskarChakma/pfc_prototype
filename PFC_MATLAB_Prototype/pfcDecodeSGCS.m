function s = pfcDecodeSGCS(dec, Z, Re, Im, opts)
%PFCDECODESGCS Per-sample SGCS of a decoder applied to latents Z, against targets (Re, Im).
%   s = pfcDecodeSGCS(dec, Z, Re, Im) returns a 1-by-N single row vector.
%   Z is D-by-N; Re and Im are 32-by-13-by-N.

arguments
    dec struct
    Z (:,:) single
    Re (:,:,:) single
    Im (:,:,:) single
    opts.BatchSize (1,1) double = 2000
    opts.UseGPU (1,1) logical = true
end
useGPU = opts.UseGPU && canUseGPU;
N = size(Z, 2);
s = zeros(1, N, "single");
for k = 1:opts.BatchSize:N
    idx = k:min(k + opts.BatchSize - 1, N);
    zb = Z(:, idx); rb = Re(:, :, idx); ib = Im(:, :, idx);
    if useGPU, zb = gpuArray(zb); rb = gpuArray(rb); ib = gpuArray(ib); end
    v = pfcSGCS(pfcForward(dec, zb), rb, ib);
    s(idx) = gather(extractdata(v));
end
end
