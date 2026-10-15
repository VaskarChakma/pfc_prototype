function [accept, report] = pfcCertify(dS, delta, alpha, j, numPartners)
%PFCCERTIFY Partner-free compatibility certification of a model update (Theorem 1).
%   [accept, report] = pfcCertify(dS, delta, alpha, j, numPartners)
%   dS          N-by-K per-channel SGCS change (candidate minus old) for K certifiers trained on
%               the exchanged dataset, on N held-out channels independent of the update
%   delta       tolerated SGCS loss (e.g., 0.01)
%   alpha       channel-sampling risk (e.g., 0.05), split as alpha/K per certifier
%   j           order statistic (default 1 = most pessimistic certifier); 1 <= j <= K
%   numPartners optional M, to report the family-wise bound (j = 1)
%
%   Rule: accept iff the j-th smallest one-sided empirical-Bernstein bound L_k >= -delta.
%   Guarantee (exchangeable partner and certifiers, update independent of certifiers/test channels):
%     per partner:              P(accept and partner change < -delta) <= j/(K+1) + alpha
%     all M partners (j = 1):   P(accept and some partner broken)    <= M/(K+M) + alpha

arguments
    dS (:,:) double
    delta (1,1) double {mustBePositive} = 0.01
    alpha (1,1) double {mustBeInRange(alpha, 0, 1)} = 0.05
    j (1,1) double {mustBeInteger, mustBePositive} = 1
    numPartners (1,1) double {mustBeInteger, mustBeNonnegative} = 0
end
[N, K] = size(dS);
if j > K
    error("pfcCertify:order", "j (%d) must not exceed the number of certifiers K (%d).", j, K);
end
if N < 2
    error("pfcCertify:samples", "At least two test channels are needed.");
end
R = 2;                                    % paired SGCS differences lie in [-1, 1]
a = alpha / K;
m = mean(dS, 1);
v = var(dS, 0, 1);
lcb = m - sqrt(2 * v * log(2 / a) / N) - 7 * R * log(2 / a) / (3 * (N - 1));   % Maurer & Pontil (2009), Thm 4
s = sort(lcb);
accept = s(j) >= -delta;
report = struct(mean=m, lcb=lcb, decisionStatistic=s(j), riskBoundPerPartner=j / (K + 1) + alpha, ...
                riskBoundAllPartners=NaN, N=N, K=K);
if numPartners > 0 && j == 1
    report.riskBoundAllPartners = numPartners / (K + numPartners) + alpha;
end
end
