function Zq = pfcQuant(Z)
%PFCQUANT 2-bit scalar quantizer on tanh(Z) with a straight-through gradient.
%   Zq = pfcQuant(Z) returns tanh(Z) snapped to the levels {-0.75,-0.25,0.25,0.75}
%   in the forward pass; gradients flow through tanh only (works inside dlfeval).
T = tanh(Z);
t = extractdata(T);
q = -1 + 0.5 * (min(max(floor((t + 1) / 0.5), 0), 3) + 0.5);
Zq = T + (q - t);          % numeric offset carries no gradient
end
