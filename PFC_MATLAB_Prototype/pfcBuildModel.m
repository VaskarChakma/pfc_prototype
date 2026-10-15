function model = pfcBuildModel(kind, role, latentDim, opts)
%PFCBUILDMODEL Encoder or decoder of one model family as a dlnetwork.
%   model = pfcBuildModel(kind, role, D) with kind "mlp" | "cnn" | "attn" and role
%   "enc" (832 features -> D pre-tanh latents) or "dec" (D latents -> 832 angle-delay outputs).
%   Options: Width (MLP hidden sizes), Channels and Hidden (CNN), Dim, Heads and Blocks
%   (attention), Seed.
%   The attention family uses one token per subband, a learned position embedding and
%   pre-norm transformer blocks with residual connections.
%   Requires Deep Learning Toolbox R2023b or later.

arguments
    kind (1,1) string {mustBeMember(kind, ["mlp", "cnn", "attn"])}
    role (1,1) string {mustBeMember(role, ["enc", "dec"])}
    latentDim (1,1) double = 64
    opts.Width (1,:) double = [512 512]
    opts.Channels (1,1) double = 12
    opts.Hidden (1,1) double = 256
    opts.Dim (1,1) double = 64
    opts.Heads (1,1) double = 4
    opts.Blocks (1,1) double = 2
    opts.Seed (1,1) double = 0
end
nSub = 13; nTx = 32; nIn = 2 * nSub * nTx;
rng(opts.Seed, "twister");

switch kind
    case "mlp"
        if role == "enc", inSize = nIn; outSize = latentDim; else, inSize = latentDim; outSize = nIn; end
        layers = featureInputLayer(inSize, Normalization="none");
        for w = opts.Width
            layers = [layers; fullyConnectedLayer(w); geluLayer]; %#ok<AGROW>
        end
        net = dlnetwork([layers; fullyConnectedLayer(outSize)]);

    case "cnn"
        c = opts.Channels;
        if role == "enc"
            layers = [imageInputLayer([nSub nTx 2], Normalization="none")
                      convolution2dLayer(3, c, Padding="same"); geluLayer
                      convolution2dLayer(3, c, Padding="same"); geluLayer
                      averagePooling2dLayer([1 4], Stride=[1 4])
                      fullyConnectedLayer(opts.Hidden); geluLayer
                      fullyConnectedLayer(latentDim)];
        else
            layers = [featureInputLayer(latentDim, Normalization="none")
                      fullyConnectedLayer(nSub * (nTx/4) * c); geluLayer
                      functionLayer(@(x) dlarray(reshape(stripdims(x), nSub, nTx/4, c, []), "SSCB"), ...
                                    Formattable=true, Name="toImage")
                      transposedConv2dLayer([1 4], c, Stride=[1 4])
                      convolution2dLayer(3, c, Padding="same"); geluLayer
                      convolution2dLayer(3, c, Padding="same"); geluLayer
                      convolution2dLayer(3, 2, Padding="same")];
        end
        net = dlnetwork(layers);

    case "attn"
        dm = opts.Dim;
        % No input layer: the network is initialized from an example input below.
        if role == "enc"
            lg = layerGraph(fullyConnectedLayer(dm, Name="tok"));
            example = dlarray(zeros(2 * nTx, 1, nSub, "single"), "CBT");
        else
            lg = layerGraph([fullyConnectedLayer(nSub * dm, Name="lift")
                             functionLayer(@(x) dlarray(permute(reshape(stripdims(x), dm, nSub, []), [1 3 2]), "CBT"), ...
                                           Formattable=true, Name="tok")]);
            example = dlarray(zeros(latentDim, 1, "single"), "CB");
        end
        lg = addLayers(lg, positionEmbeddingLayer(dm, nSub, Name="pos"));
        lg = addLayers(lg, additionLayer(2, Name="addpos"));
        lg = connectLayers(lg, "tok", "pos");
        lg = connectLayers(lg, "tok", "addpos/in1");
        lg = connectLayers(lg, "pos", "addpos/in2");
        x = "addpos";
        for b = 1:opts.Blocks
            p = "b" + b + "_";
            lg = addLayers(lg, [layerNormalizationLayer(Name=p + "ln1")
                                selfAttentionLayer(opts.Heads, dm, Name=p + "att")]);
            lg = addLayers(lg, additionLayer(2, Name=p + "add1"));
            lg = connectLayers(lg, x, p + "ln1");
            lg = connectLayers(lg, p + "att", p + "add1/in1");
            lg = connectLayers(lg, x, p + "add1/in2");
            lg = addLayers(lg, [layerNormalizationLayer(Name=p + "ln2")
                                fullyConnectedLayer(2 * dm, Name=p + "f1")
                                geluLayer(Name=p + "g")
                                fullyConnectedLayer(dm, Name=p + "f2")]);
            lg = addLayers(lg, additionLayer(2, Name=p + "add2"));
            lg = connectLayers(lg, p + "add1", p + "ln2");
            lg = connectLayers(lg, p + "f2", p + "add2/in1");
            lg = connectLayers(lg, p + "add1", p + "add2/in2");
            x = p + "add2";
        end
        if role == "enc"
            lg = addLayers(lg, [functionLayer(@(x) dlarray(reshape(permute(stripdims(x), [1 3 2]), [], size(x, 2)), "CB"), ...
                                              Formattable=true, Name="flat")
                                fullyConnectedLayer(latentDim, Name="out")]);
            lg = connectLayers(lg, x, "flat");
        else
            lg = addLayers(lg, fullyConnectedLayer(2 * nTx, Name="out"));
            lg = connectLayers(lg, x, "out");
        end
        net = dlnetwork(lg, Initialize=false);
        net = initialize(net, example);
end
model = struct(net=net, kind=kind, role=role);
end
