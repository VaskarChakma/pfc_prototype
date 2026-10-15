function [W, profileUsed] = pfcCDLDataset(numSamples, scenario, seed)
%PFCCDLDATASET Rank-1 subband precoders from 3GPP TR 38.901 CDL channels (5G Toolbox).
%   [W, profileUsed] = pfcCDLDataset(N, scenario, seed) returns W of size 13-by-32-by-N
%   (subbands x gNB antennas x drops). Each subband vector is the channel at the centre
%   subcarrier of the subband, normalized to unit norm (the MRT precoder for one UE antenna).
%
%   scenario "S0": CDL-C, delay spread U(100,300) ns, array bearing U(-60,60) deg
%   scenario "S1": CDL-A, CDL-B or CDL-D, delay spread U(30,100) ns, bearing U(-30,90) deg
%
%   Each drop draws a new delay spread, a new array bearing (TransmitArrayOrientation,
%   TR 38.901 Sec. 7.1.3) and a new random seed for the ray phases and ray coupling.
%   Carrier: 52 RB at 30 kHz SCS, 3.5 GHz. 13 subbands of 4 RB.
%   gNB: 32-element horizontal ULA, half-wavelength spacing, one polarization. UE: one antenna.
%
%   Uses ChannelResponseOutput="ofdm-response" on R2024b or later. On older releases it uses
%   path gains with nrPerfectChannelEstimate. Uses parfor when Parallel Computing Toolbox
%   is available (otherwise parfor runs as a normal loop).

arguments
    numSamples (1,1) double {mustBePositive, mustBeInteger}
    scenario (1,1) string {mustBeMember(scenario, ["S0", "S1"])}
    seed (1,1) double = 1
end

switch scenario
    case "S0"
        profiles = "CDL-C"; dsRange = [100e-9 300e-9]; offRange = [-60 60];
    case "S1"
        profiles = ["CDL-A" "CDL-B" "CDL-D"]; dsRange = [30e-9 100e-9]; offRange = [-30 90];
end

% Draw all per-drop parameters first, so results do not depend on the number of workers.
rng(seed, "twister");
pIdx = randi(numel(profiles), numSamples, 1);
ds = dsRange(1) + diff(dsRange) * rand(numSamples, 1);
bearing = offRange(1) + diff(offRange) * rand(numSamples, 1);
chSeed = randi(2^31 - 2, numSamples, 1);

chunk = 500;
nChunks = ceil(numSamples / chunk);
parts = cell(nChunks, 1);
parfor c = 1:nChunks
    idx = (c - 1) * chunk + 1 : min(c * chunk, numSamples);
    parts{c} = generateChunk(profiles, pIdx(idx), ds(idx), bearing(idx), chSeed(idx));
end
W = cat(3, parts{:});
profileUsed = profiles(pIdx).';
end

function W = generateChunk(profiles, pIdx, ds, bearing, chSeed)
nSub = 13; rbPerSub = 4; nTx = 32;
carrier = nrCarrierConfig(SubcarrierSpacing=30, NSizeGrid=52);
ofdmInfo = nrOFDMInfo(carrier);
useOfdmResponse = ~isMATLABReleaseOlderThan("R2024b");
centreSc = ((1:nSub) - 1) * 12 * rbPerSub + 12 * rbPerSub / 2;   % centre subcarrier of each subband

chans = cell(1, numel(profiles));
for p = 1:numel(profiles)
    chans{p} = makeCDL(profiles(p), nTx, ofdmInfo, useOfdmResponse);
end

n = numel(pIdx);
W = complex(zeros(nSub, nTx, n, "single"));
for i = 1:n
    ch = chans{pIdx(i)};
    release(ch);
    ch.DelaySpread = ds(i);
    ch.Seed = chSeed(i);
    ch.TransmitArrayOrientation = [bearing(i); 0; 0];
    if useOfdmResponse
        H = ch(carrier);                                   % subcarriers x symbols x Nr x Nt
    else
        [pathGains, sampleTimes] = ch();
        pathFilters = getPathFilters(ch);
        offset = nrPerfectTimingEstimate(pathGains, pathFilters);
        H = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
    end
    assert(size(H, 1) == 12 * carrier.NSizeGrid && size(H, 4) == nTx, ...
        "pfcCDLDataset:size", "Unexpected channel response size %s.", mat2str(size(H)));
    Hs = reshape(H(centreSc, 1, 1, :), nSub, nTx);         % first OFDM symbol, one UE antenna
    W(:, :, i) = single(Hs ./ vecnorm(Hs, 2, 2));
end
end

function ch = makeCDL(profile, nTx, ofdmInfo, useOfdmResponse)
ch = nrCDLChannel;
ch.DelayProfile = profile;
ch.CarrierFrequency = 3.5e9;
ch.MaximumDopplerShift = 0;
ch.SampleRate = ofdmInfo.SampleRate;
ch.TransmitAntennaArray.Size = [1 nTx 1 1 1];          % horizontal ULA, one polarization
ch.TransmitAntennaArray.ElementSpacing = [0.5 0.5 1 1];
ch.TransmitAntennaArray.Element = "isotropic";
ch.ReceiveAntennaArray.Size = [1 1 1 1 1];
ch.ReceiveAntennaArray.Element = "isotropic";
ch.RandomStream = "mt19937ar with seed";
ch.ChannelFiltering = false;
ch.NumTimeSamples = sum(ofdmInfo.SymbolLengths(1:ofdmInfo.SymbolsPerSlot));
if useOfdmResponse
    ch.ChannelResponseOutput = "ofdm-response";
end
end
