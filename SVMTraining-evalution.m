
%% Set up our training data
allImages = imageDatastore('DFU224', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allImages.ReadFcn = @customReadDatastoreImage;
[trainingImages, testImages] = splitEachLabel(allImages, 0.60, 'randomize');
%% Measure network accuracy
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
%% confusion metrix 
% RE= testImages.Labels;
cm = confusionmat(YTest, YPred)
[cm,order] = confusionmat(YTest, YPred,'Order',{'Circular','Elongated','Other'}) 
cm1= bsxfun (@rdivide, cm, sum(cm,2))
mean(diag(cm1))
%%
cm2 = confusionchart(YTest, YPred);
%%
cm4 = confusionchart(cm1);
%%
tp_m = diag(cm);

 for i = 1:3 
    TP = tp_m(i);
    FP = sum(cm(:, i), 1) - TP;
    FN = sum(cm(i, :), 2) - TP;
    TN = sum(cm(:)) - TP - FP - FN;

    Accuracy = (TP+TN)./(TP+FP+TN+FN);

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FScore = (2*(PPV * TPR)) / (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end
end
%% https://www.mathworks.com/help/deeplearning/ref/predict.html
YPred = predict(myNet,testImages)
YPred(1:2,:)

%%
idx = [1 27 20 4 5 41 6 9 18 43 3 16];
figure
for i = 1:numel(idx)
    subplot(3,4,i)
    I = readimage(testImages,idx(i));
    testImages.ReadFcn = @customReadDatastoreImage;
    label = predictedLabels(idx(i));
   % prob = num2str(100*max(YPred(idx(i),:)),2);
    imshow(I)
    title(char(label))
end
%%
function data=customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end