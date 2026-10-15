function summary = pfcAnalyzeStress(outNW, outUE, outDir)
%PFCANALYZESTRESS Summarize both sides of the stress study.
%   summary = pfcAnalyzeStress(outNW, outUE, outDir) takes the outputs of runPFCStress("nw")
%   and runPFCStress("ue"), prints the decision table and saves stress_summary.json.
%   An update is "safe" if no hidden partner (standard or stress) loses more than delta on the
%   independent truth channels.
%   Rule baselines:
%     provenance  : same dataset, same pairing ID, so every update keeps the ID
%     always      : re-pair after every update
%     certification: pfcCertify (j = 1)
%   Rank event (Theorem 1): a standard partner's true change is below the lowest true change of
%   all K certifiers. Theorem 1 predicts a rate of 1/(K+1) for exchangeable partners.

outs = {outNW, outUE};
t = struct(updates=0, accepted=0, wrongKeepCert=0, wrongKeepCertStd=0, wrongKeepCertStress=0, ...
           cautiousReject=0, wrongKeepProvenance=0, needlessAlways=0, ...
           partnerChecksAcceptedStd=0, brokenAcceptedStd=0, partnerChecksAcceptedStress=0, brokenAcceptedStress=0, ...
           rankEventsStd=0, rankCasesStd=0, rankEventsStress=0, rankCasesStress=0);
rows = struct([]);
cautious = strings(0);
for o = 1:2
    out = outs{o};
    d = out.delta;
    roles = out.roles(:).';
    isStd = roles == 0; isCert = roles == 1; isStress = roles == 2;
    cands = out.cands;
    for c = 1:numel(cands)
        r = cands(c);
        truth = r.truth(:).';
        worst = min(r.worstStd, r.worstStress);
        safe = worst >= -d;
        t.updates = t.updates + 1;
        t.accepted = t.accepted + r.accept;
        t.wrongKeepCert = t.wrongKeepCert + (r.accept && ~safe);
        t.wrongKeepCertStd = t.wrongKeepCertStd + (r.accept && r.worstStd < -d);
        t.wrongKeepCertStress = t.wrongKeepCertStress + (r.accept && r.worstStress < -d);
        t.cautiousReject = t.cautiousReject + (~r.accept && safe);
        t.wrongKeepProvenance = t.wrongKeepProvenance + ~safe;
        t.needlessAlways = t.needlessAlways + safe;
        if r.accept
            t.partnerChecksAcceptedStd = t.partnerChecksAcceptedStd + sum(isStd);
            t.brokenAcceptedStd = t.brokenAcceptedStd + r.brokenStd;
            t.partnerChecksAcceptedStress = t.partnerChecksAcceptedStress + sum(isStress);
            t.brokenAcceptedStress = t.brokenAcceptedStress + r.brokenStress;
        end
        if ~r.accept && safe
            cautious(end + 1) = sprintf("%s/%s (true worst %+.4f)", out.side, r.name, worst); %#ok<AGROW>
        end
        lowCert = min(truth(isCert));
        t.rankEventsStd = t.rankEventsStd + sum(truth(isStd) < lowCert);
        t.rankCasesStd = t.rankCasesStd + sum(isStd);
        t.rankEventsStress = t.rankEventsStress + sum(truth(isStress) < lowCert);
        t.rankCasesStress = t.rankCasesStress + sum(isStress);
        partners = isStd | isStress;
        s1 = r.s1(:).'; s1Old = r.s1Old(:).';
        rows = [rows, struct(side=string(out.side), name=string(r.name), accept=logical(r.accept), ...
            certificate=r.certMinLcb, trueWorst=worst, ...
            partnerDriftSGCSOld=mean(s1Old(partners)), partnerDriftSGCSNew=mean(s1(partners)))]; %#ok<AGROW>
    end
end
K = outNW.K;
summary.totals = t;
summary.rankRateStd = t.rankEventsStd / max(t.rankCasesStd, 1);
summary.rankRateStress = t.rankEventsStress / max(t.rankCasesStress, 1);
summary.rankRatePredicted = 1 / (K + 1);
summary.cautiousRejections = cautious;
summary.ownPairSGCS = [outNW.ownPairSGCS, outUE.ownPairSGCS];
acc = rows([rows.accept]);
if ~isempty(acc)
    [~, i] = max([acc.partnerDriftSGCSNew] - [acc.partnerDriftSGCSOld]);
    summary.bestCertifiedDriftGain = acc(i);
end
fx = rows(startsWith([rows.name], "fixed_point"));
summary.fixedPoint = fx;
summary.rows = rows;

fprintf("\n===== Stress study summary (truth on independent channels, delta = %.3f) =====\n", outNW.delta);
fprintf("updates %d, accepted %d\n", t.updates, t.accepted);
fprintf("certification: wrong keeps %d (standard %d, stress %d), cautious rejections %d\n", ...
        t.wrongKeepCert, t.wrongKeepCertStd, t.wrongKeepCertStress, t.cautiousReject);
fprintf("broken partner checks among accepted updates: standard %d of %d, stress %d of %d\n", ...
        t.brokenAcceptedStd, t.partnerChecksAcceptedStd, t.brokenAcceptedStress, t.partnerChecksAcceptedStress);
fprintf("provenance rule (same dataset, same ID): wrong keeps %d\n", t.wrongKeepProvenance);
fprintf("always re-pair: needless re-pairings %d\n", t.needlessAlways);
fprintf("rank event: standard %.1f%% of %d (Theorem 1: %.2f%%), stress %.1f%% of %d\n", ...
        100 * summary.rankRateStd, t.rankCasesStd, 100 / (K + 1), 100 * summary.rankRateStress, t.rankCasesStress);
for k = 1:numel(cautious), fprintf("cautious rejection: %s\n", cautious(k)); end
if isfield(summary, "bestCertifiedDriftGain")
    b = summary.bestCertifiedDriftGain;
    fprintf("largest drift gain among accepted updates: %s/%s, partner SGCS on S1 %.3f -> %.3f, true worst on S0 %+.4f\n", ...
            b.side, b.name, b.partnerDriftSGCSOld, b.partnerDriftSGCSNew, b.trueWorst);
end
for k = 1:numel(fx)
    fprintf("%s %-18s accept=%d true worst %+.4f\n", fx(k).side, fx(k).name, fx(k).accept, fx(k).trueWorst);
end
if nargin >= 3
    writelines(jsonencode(summary, PrettyPrint=true), fullfile(outDir, "stress_summary.json"));
end
end
