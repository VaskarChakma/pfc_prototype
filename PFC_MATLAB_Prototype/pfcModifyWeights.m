function model = pfcModifyWeights(model, kind, value, seed)
%PFCMODIFYWEIGHTS Model updates that change weights directly.
%   model = pfcModifyWeights(model, "noise", s, seed)  adds Gaussian noise with standard deviation
%           s times the standard deviation of each weight tensor (an imprecise retraining).
%   model = pfcModifyWeights(model, "prune", f)        sets the fraction f of smallest-magnitude
%           entries of every weight matrix or kernel to zero (vectors such as biases are kept).
%   model = pfcModifyWeights(model, "fixed", bits)     symmetric per-tensor fixed-point rounding
%           of every learnable tensor to the given word length (deployment on embedded hardware).

arguments
    model struct
    kind (1,1) string {mustBeMember(kind, ["noise", "prune", "fixed"])}
    value (1,1) double
    seed (1,1) double = 0
end
rng(seed, "twister");
switch kind
    case "noise"
        f = @(w) w + value * std(w(:)) * randn(size(w), "like", w);
    case "prune"
        f = @(w) pruneTensor(w, value);
    case "fixed"
        L = 2^(value - 1) - 1;
        f = @(w) round(w ./ (max(abs(w), [], "all") + 1e-12) * L) / L .* (max(abs(w), [], "all") + 1e-12);
end
model.net = dlupdate(@(w) dlarray(f(gather(extractdata(w)))), model.net);
end

function w = pruneTensor(w, frac)
if isvector(w), return, end
a = sort(abs(w(:)));
th = a(max(1, min(numel(a), round(frac * numel(a)))));
w(abs(w) < th) = 0;
end
