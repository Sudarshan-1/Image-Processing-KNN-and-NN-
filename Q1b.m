close all;
clear all;
clc;


%% Add MatConvNet to the project
run './matconvnet/matlab/vl_setupnn'

% load the 227MB pre-trained CNN
net = load('imagenet-caffe-alex.mat') ;


%% Preparing testing data
disp('Preparing testing data');
folderCat = './DogCat/Testing/Cat/';
folderDog = './DogCat/Testing/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));


for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    im = imread([folderCat filename]);
    im = imresize(im,[227 227]);
    im_ = single(im) ;
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    for j = 1:3
        im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
    end
    % run the CNN
    res = vl_simplenn(net, im_) ;
    % visualize the classification result
    scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
    figure ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{best}, best, bestScore)) ;
end



for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    im = imread([folderDog filename]);
    im = imresize(im,[227 227]);
    im_ = single(im) ;
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    for j = 1:3
        im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
    end
    % run the CNN
    res = vl_simplenn(net, im_) ;
    % visualize the classification result
    scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
    figure ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{best}, best, bestScore)) ;
end


