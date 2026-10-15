function Z = pfcEncode(enc, X, opts)
%PFCENCODE Run an encoder in batches and return latents as a numeric D-by-N single array.
%   Z = pfcEncode(enc, X) returns the 2-bit quantized latents (what the UE sends).
%   Z = pfcEncode(enc, X, Output="float") returns tanh of the pre-quantization latents
%   (the latent "before quantization" that Option 4-1 dataset exchange shares).

arguments
    enc struct
    X (:,:) single
    opts.Output (1,1) string {mustBeMember(opts.Output, ["quant", "float"])} = "quant"
    opts.BatchSize (1,1) double = 2000
    opts.UseGPU (1,1) logical = true
end
useGPU = opts.UseGPU && canUseGPU;
N = size(X, 2);
Z = zeros(0, N, "single");
for s = 1:opts.BatchSize:N
    idx = s:min(s + opts.BatchSize - 1, N);
    xb = X(:, idx);
    if useGPU, xb = gpuArray(xb); end
    y = predictLatent(enc, xb, opts.Output);
    if isempty(Z), Z = zeros(size(y, 1), N, "single"); end
    Z(:, idx) = gather(y);
end
end

function y = predictLatent(enc, xb, output)
pre = pfcForward(enc, xb);
if output == "float"
    y = extractdata(tanh(pre));
else
    y = extractdata(pfcQuant(pre));
end
y = single(y);
end
