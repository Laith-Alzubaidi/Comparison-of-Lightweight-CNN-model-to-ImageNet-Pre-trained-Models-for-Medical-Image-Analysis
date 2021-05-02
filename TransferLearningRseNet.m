
%%
net = resnet50;
lgraph = layerGraph(net);
net.Layers

analyzeNetwork(lgraph)
%% Set up our training data
allImages = imageDatastore('SickleCells80', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allImages.ReadFcn = @customReadDatastoreImage;
[trainingImages, testImages] = splitEachLabel(allImages, 0.80, 'randomize');
%%
lgraph = removeLayers(lgraph,{'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});

numClasses = numel(categories(trainingImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc21','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','New_softmax1')
    classificationLayer('Name','ClassificationLayer_1')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'avg_pool','fc21');
analyzeNetwork(lgraph)

%

%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 75, 'MiniBatchSize', 16,'Plots','training-progress');
myNet = trainNetwork(trainingImages, lgraph, opts);

%% Measure network accuracy
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
%%
save ResNetSickle myNet

function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end