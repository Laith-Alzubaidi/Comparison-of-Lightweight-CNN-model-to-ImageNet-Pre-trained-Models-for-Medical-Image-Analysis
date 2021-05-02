net = myNet;
net.Layers
inputSize = net.Layers(1).InputSize;
%%
imds =imageDatastore('SickleCells80', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testImages.ReadFcn = @customReadDatastoreImage;
[imdsTrain,imdsTest] = splitEachLabel(imds,0.70,'randomized');

%%
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
%%
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc2';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
%%
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
%%
idx = [76 4 78 79 80  82 209 210 211  213 214 215 333 334 335 336 338 339 5 170];
figure
for i = 1:numel(idx)
    subplot(4,5,i)
    I = readimage(imdsTest,idx(i));
    testImages.ReadFcn = @customReadDatastoreImage;
    label = YPred(idx(i));
   % prob = num2str(100*max(YPred(idx(i),:)),2);
    imshow(I)
    title(char(label))
end
% title('predicted labels')
%%
figure
for i = 1:numel(idx)
    subplot(3,3,i)
    imshow(XValidation(:,:,:,idx(i)));
    prob = num2str(100*max(YPred(idx(i),:)),3);
    predClass = char(YValPred(idx(i)));
    title([predClass,', ',prob,'%'])
end

%%
accuracy = mean(YPred == YTest)

%%
function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[80 80]);
end

