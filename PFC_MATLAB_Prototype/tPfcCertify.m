classdef tPfcCertify < matlab.unittest.TestCase
    %TPFCCERTIFY Requirement-style tests for the certification rule (base MATLAB only).
    %   REQ-1 An update that does not change any certifier is accepted.
    %   REQ-2 An update that degrades every certifier by 2*delta is rejected.
    %   REQ-3 A single badly degraded certifier blocks acceptance at j = 1, not at j = 2.
    %   REQ-4 The empirical-Bernstein bound covers the true mean at the stated level.
    %   REQ-5 The reported risk bounds equal j/(K+1)+alpha and M/(K+M)+alpha.
    %   REQ-6 Rank term: with one fragile implementation among K+1 exchangeable ones, the
    %         accept-and-broken rate does not exceed 1/(K+1)+alpha (up to Monte Carlo error).
    %   REQ-7 j > K is rejected as an input error.

    methods (TestMethodSetup)
        function seed(~)
            rng(1, "twister");
        end
    end

    methods (Test)
        function acceptsNoChange(tc)
            tc.verifyTrue(pfcCertify(1e-3 * randn(20000, 9), 0.01, 0.05));
        end

        function rejectsUniformDegradation(tc)
            tc.verifyFalse(pfcCertify(-0.02 + 1e-3 * randn(20000, 9), 0.01, 0.05));
        end

        function oneBadCertifierBlocks(tc)
            dS = 1e-3 * randn(20000, 9);
            dS(:, 4) = dS(:, 4) - 0.05;
            tc.verifyFalse(pfcCertify(dS, 0.01, 0.05, 1));
            tc.verifyTrue(pfcCertify(dS, 0.01, 0.05, 2));
        end

        function bernsteinCoverage(tc)
            trials = 400; misses = 0; K = 5; alpha = 0.05; mu = -0.004;
            for t = 1:trials
                dS = max(min(mu + 0.03 * randn(3000, K), 1), -1);
                [~, r] = pfcCertify(dS, 0.01, alpha);
                misses = misses + any(r.lcb > mu);
            end
            tc.verifyLessThanOrEqual(misses / trials, alpha);
        end

        function riskBoundsReported(tc)
            [~, r] = pfcCertify(zeros(100, 15), 0.01, 0.05, 1, 8);
            tc.verifyEqual(r.riskBoundPerPartner, 1/16 + 0.05, AbsTol=1e-12);
            tc.verifyEqual(r.riskBoundAllPartners, 8/23 + 0.05, AbsTol=1e-12);
        end

        function rankTermHolds(tc)
            K = 9; trials = 1000; bad = 0; N = 20000;   % N large enough that the range term is below delta
            for t = 1:trials
                mu = zeros(1, K + 1); mu(randi(K + 1)) = -0.05;      % one fragile implementation
                dS = mu(1:K) + 0.02 * randn(N, K);                % certifiers = first K, partner = last
                bad = bad + (pfcCertify(dS, 0.01, 0.05) && mu(K + 1) < -0.01);
            end
            se = sqrt(0.1 * 0.9 / trials);
            tc.verifyLessThanOrEqual(bad / trials, 1/(K + 1) + 0.05 + 3 * se);
            tc.verifyGreaterThan(bad / trials, 0.05);   % non-vacuous: the rank term is actually reached
        end

        function orderStatisticValidated(tc)
            tc.verifyError(@() pfcCertify(zeros(10, 3), 0.01, 0.05, 4), "pfcCertify:order");
        end
    end
end
