clear all;
close all;
clc;

%% Add MatConvNet to the project
run './matconvnet/matlab/vl_setupnn'

% load the 227MB pre-trained CNN
net = load('imagenet-caffe-alex.mat') ;


%% Preparing training data
disp('Preparing training data');
folderCat = './DogCats/Training/Cat/';
folderDog = './DogCats/Training/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

Catfeat=zeros(length(filesCat)+length(filesDog),1);
Dogfeat=zeros(length(filesCat)+length(filesDog),1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    im=img(:,end:-1:1,:);
    im=imresize(im,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
    Sum_b=J(49,49)-Sum_w;
    haar1=Sum_w-Sum_b;
    Catfeat(i)=haar1;
    im=imresize(img,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
    Sum_b=J(49,49)-Sum_w;
    haar2=Sum_w-Sum_b;
    Catfeat(i+length(filesCat))=haar2;
    
    
end

for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    im=img(:,end:-1:1,:);
    im=imresize(im,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
    Sum_b=J(49,49)-Sum_w;
    haar1=Sum_w-Sum_b;
    Dogfeat(i)=haar1;
    im=imresize(img,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
    Sum_b=J(49,49)-Sum_w;
    haar2=Sum_w-Sum_b;
    Dogfeat(i+length(filesDog))=haar2;

    
    
end


%% Preparing the testing data

disp('Preparing the testing Data');
folderCatTest='./DogCats/Testing/Cat/';
folderDogTest='./Dogcats/Testing/Dog/';

filesCatTest=dir(fullfile(folderCatTest,'*.jpg'));
filesDogTest=dir(fullfile(folderDogTest,'*.jpg'));

featsTest = zeros(length(filesCatTest) + length(filesDogTest), 1);
groundtruthLabel=zeros(length(filesCatTest)+length(filesDogTest),1);
predictedLabel = zeros(length(filesCatTest) + length(filesDogTest), 1);



m=zeros(5,1);

for x=1:length(m)

     k=input("enter the value of k ");

    for i = 1:length(filesCatTest)
        
        disp(i);
        filename = filesCatTest(i,1).name;
        img = imread([folderCatTest filename]);
        im=imresize(img,[128 128]);
        J = integralImage(im);
        Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
        Sum_b=J(49,49)-Sum_w;
        haar=Sum_w-Sum_b;
        dist=distChiSq(haar,Catfeat);
        [value,idx] = sort(dist);
        value(value < 0) = 0;
        value(value ~= 0) = 1;
        value = value(1,1:k);
        result = mode(value);      
        groundtruthLabel(i)=0;
        predictedLabel(i)=result;
    end

    for i = 1:length(filesDogTest)
        disp(i);
        filename = filesDogTest(i,1).name;
        img = imread([folderDogTest filename]);
        im=imresize(img,[128 128]);
        J = integralImage(im);
        Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
        Sum_b=J(49,49)-Sum_w;
        haar=Sum_w-Sum_b;
        
        dist=distChiSq(haar,Dogfeat);
        [value,idx] = sort(dist);
        value(value < 0) = 1;
        value(value ~= 0) = 0;
        value = value(1,1:k);
        result = mode(value);      
        groundtruthLabel(i+length(folderCatTest))=1;
        predictedLabel(i + length(folderCatTest))=result;
    end





    accurate=0;
    for i=1:length(predictedLabel)
    
        if(predictedLabel(i)==groundtruthLabel(i))
            accurate=accurate+1;
        end
    end

    accuracy = accurate/length(groundtruthLabel);
    disp(['The accuracy:' num2str(accuracy * 100) '%']);


    
end     










