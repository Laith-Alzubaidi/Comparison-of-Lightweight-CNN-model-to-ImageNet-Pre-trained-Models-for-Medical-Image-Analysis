
netWidth = 32;
layers = [
    imageInputLayer([224 224 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
   convolution2dLayer(5,netWidth,'Padding','same','Stride',2,'Name','ConvQ1')
    batchNormalizationLayer('Name','BNQ1')
    reluLayer('Name','reluQ1')
    
    %part1
     convolution2dLayer(1,netWidth,'Padding','same','Stride',1,'Name','ConvQ111')
    batchNormalizationLayer('Name','BNQ111')
    reluLayer('Name','reluQ111')
    
   depthConcatenationLayer(5,'Name','concat_1')
   batchNormalizationLayer('Name','BNQ113')
   %part2
   convolution2dLayer(1,2*netWidth,'Padding','same','Stride',2,'Name','ConvQ1111')
    batchNormalizationLayer('Name','BNQ1111')
    reluLayer('Name','reluQ1111')
    
   depthConcatenationLayer(5,'Name','concat_11')
   batchNormalizationLayer('Name','BNQ117')
   
    
   globalAveragePooling2dLayer('Name','GlobalAvergePool')
 %    averagePooling2dLayer(20,'Stride',1,'Name','avergePool')
    fullyConnectedLayer(200,'Name','fc1')
     dropoutLayer('Name','drop1')
    fullyConnectedLayer(2,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];
    
%%
lgraph = layerGraph(layers);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);

plot(lgraph);
%% part1
Skip1= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','skipConvQ1')
    batchNormalizationLayer('Name','skipBNQ1')
    reluLayer('Name','skipreluQ1')]
   
lgraph = addLayers(lgraph,Skip1);
lgraph = connectLayers(lgraph,'reluQ1','skipConvQ1');
lgraph = connectLayers(lgraph,'skipreluQ1','concat_1/in2');

%%
skip2 = [
    convolution2dLayer(5,netWidth,'Padding','same','Stride',1,'Name','skipConvU1')
    batchNormalizationLayer('Name','skipBNU1')
    reluLayer('Name','skipreluU1')]

lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'reluQ1','skipConvU1');
lgraph = connectLayers(lgraph,'skipreluU1','concat_1/in3');

%%

skip3 = [
    convolution2dLayer(7,netWidth,'Padding','same','Stride',1,'Name','skipConvT1')
    batchNormalizationLayer('Name','skipBNT1')
    reluLayer('Name','skipreluT1')]
   

lgraph = addLayers(lgraph,skip3);
lgraph = connectLayers(lgraph,'reluQ1','skipConvT1');
lgraph = connectLayers(lgraph,'skipreluT1','concat_1/in4');

lgraph = connectLayers(lgraph,'reluQ1','concat_1/in5');
%% part2  ////////////////////////////////////////////////////////////////
Skip11= [
    convolution2dLayer(3,2*netWidth,'Padding','same','Stride',2,'Name','skipConvQ11')
    batchNormalizationLayer('Name','skipBNQ11')
    reluLayer('Name','skipreluQ11')]
   
lgraph = addLayers(lgraph,Skip11);
lgraph = connectLayers(lgraph,'BNQ113','skipConvQ11');
lgraph = connectLayers(lgraph,'skipreluQ11','concat_11/in2');

%%
skip22 = [
    convolution2dLayer(5,2*netWidth,'Padding','same','Stride',2,'Name','skipConvU11')
    batchNormalizationLayer('Name','skipBNU11')
    reluLayer('Name','skipreluU11')]

lgraph = addLayers(lgraph,skip22);
lgraph = connectLayers(lgraph,'BNQ113','skipConvU11');
lgraph = connectLayers(lgraph,'skipreluU11','concat_11/in3');

%%

skip33 = [
    convolution2dLayer(7,2*netWidth,'Padding','same','Stride',2,'Name','skipConvT11')
    batchNormalizationLayer('Name','skipBNT11')
    reluLayer('Name','skipreluT11')]
   

lgraph = addLayers(lgraph,skip33);
lgraph = connectLayers(lgraph,'BNQ113','skipConvT11');
lgraph = connectLayers(lgraph,'skipreluT11','concat_11/in4');

%%
SkipE11z= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',2,'Name','skipConvQEEE1z')
    batchNormalizationLayer('Name','skipBNQEEE1z')
    reluLayer('Name','skipreluQ1EEEz')];
   
lgraph = addLayers(lgraph,SkipE11z);
lgraph = connectLayers(lgraph,'reluQ1','skipConvQEEE1z');
lgraph = connectLayers(lgraph,'skipreluQ1EEEz','concat_11/in5');
%%
analyzeNetwork(lgraph)

%% Set up our training data
allImages = imageDatastore('DFUtran', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allImages.ReadFcn = @customReadDatastoreImage;
[trainingImages, testImages] = splitEachLabel(allImages, 0.95, 'randomize');
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 100, 'MiniBatchSize', 16,'Plots','training-progress');
myNet = trainNetwork(trainingImages, lgraph, opts);
%% Measure network accuracy
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
%% 
save DFUNetTransfer myNet


%%
function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end
