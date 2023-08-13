close all;
clear all;
clc;


M=[];


%% for the 1st Haar feature
folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,4)-J(4,41)+J(4,4);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(i,:)=haar;
    
    disp(haar);
    
end





%% for the 2nd Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(11,5)-J(3,5)-J(11,3)+J(3,3);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(25+i,:)=haar;
    
    disp(haar);

end



%% for the 3rd Haar Feature



folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(21,41)-J(19,41)-J(21,19)+J(19,19);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(50+i,:)=haar;
    disp(haar);

end




%% for the 4th Haar Feature


folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(4,39)-J(4,25)-J(3,39)+J(3,25);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(75+i,:)=haar;
    disp(haar);

end





%% for the 5th Haar Feature



folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(46,39)-J(46,26)-J(3,39)+J(3,26);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(100+i,:)=haar;
    disp(haar);

end



%% for the 6th Haar Feature



folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(43,46)-J(3,46)-J(43,4)+J(3,4);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(125+i,:)=haar;
    disp(haar);

end




%% for the 7th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(17,22)-J(17,21)-J(16,22)+J(16,21);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(150+i,:)=haar;
    disp(haar);

end


%% for the 8th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w1=J(43,5)-J(43,3)-J(25,5)+J(25,3);
    Sum_w2=J(43,38)-J(43,36)-J(25,38)+J(25,36);
    Sum_b=J(49,49)-Sum_w1-Sum_w2;
    haar=Sum_w1+Sum_w2-Sum_b;
    M(175+i,:)=haar;
    disp(haar);

end



%% for the 9th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(41,41)-J(41,36)-J(11,41)+J(11,36);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(200+i,:)=haar;
    disp(haar);

end



%% for the 10th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(27,31)-J(27,25)-J(23,31)+J(23,25);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(225+i,:)=haar;
    disp(haar);

end


%% for the 11th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(5,47)-J(5,45)-J(3,47)+J(3,45);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(250+i,:)=haar;
    disp(haar);

end


%% for the 12th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(7,5)-J(7,2)-J(6,5)+J(6,3);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(275+i,:)=haar;
    disp(haar);

end


%% for the 13th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(25,31)-J(25,16)-J(21,31)+J(21,16);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(300+i,:)=haar;
    disp(haar);

end


%% for the 14th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(36,4)-J(36,3)-J(3,4)+J(3,3);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(325+i,:)=haar;
    disp(haar);

end



%% for the 15th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(31,13)-J(31,6)-J(3,13)+J(3,6);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(350+i,:)=haar;
    disp(haar);

end



%% for the 16th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(4,28)-J(4,27)-J(3,28)+J(3,27);
    Sum_b=J(49,49)-Sum_w/2;
    haar=Sum_w-Sum_b;
    M(375+i,:)=haar;
    disp(haar);

end



%% for the 17th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(4,29)-J(4,28)-J(3,29)+J(3,28);
    Sum_b=J(49,49)-Sum_w/4;
    haar=Sum_w-Sum_b;
    M(400+i,:)=haar;

    disp(haar);

end


%% for the 18th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(4,29)-J(4,28)-J(3,29)+J(3,28);
    Sum_b=J(49,49)-Sum_w/4;
    haar=Sum_w-Sum_b;
    M(425+i,:)=haar;
    disp(haar);

end


%% for the 19th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(37,29)-J(37,27)-J(36,29)+J(36,27);
    Sum_b=J(49,49)-Sum_w/2;
    haar=Sum_w-Sum_b;
    M(450+i,:)=haar;
    disp(haar);

end



%% for the 20th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(26,4)-J(26,3)-J(25,4)+J(25,3);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(475+i,:)=haar;
    disp(haar);

end




%% for the 21st Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(36,31)-J(36,25)-J(35,31)+J(35,25);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(500+i,:)=haar;
    disp(haar);

end


%% for the 22nd Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(47,31)-J(46,31)-J(47,27)+J(46,27);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(525+i,:)=haar;
    disp(haar);

end


%% for the 23rd Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(11,21)-J(11,6)-J(4,21)+J(4,6);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(550+i,:)=haar;
    disp(haar);

end




%% for the 24th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w1=J(25,21)-J(25,3)-J(24,21)+J(24,3);
    Sum_w2=J(25,36)-J(25,23)-J(24,36)+J(24,23);
    Sum_b=J(49,49)-(Sum_w1-Sum_w2)/2;
    haar=Sum_w-Sum_b;
    M(575+i,:)=haar;
    disp(haar);

end



%% for the 25th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(35,6)-J(35,4)-J(33,6)+J(33,3);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(600+i,:)=haar;
    disp(haar);

end



%% for the 26th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(22,5)-J(22,4)-J(21,5)+J(21,4);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(625+i,:)=haar;
    disp(haar);

end



%% for the 27th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(45,45)-J(45,16)-J(11,45)+J(11,16);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(650+i,:)=haar;
    disp(haar);

end


%% for the 28th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(20,29)-J(20,28)-J(19,29)+J(19,28);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(675+i,:)=haar;
    disp(haar);

end



%% for the 29th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(46,36)-J(46,33)-J(36,36)+J(36,33);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(700+i,:)=haar;
    disp(haar);

end





%% for the 30th Haar Feature

folderCat = './DogCat/Training/Cat/';
files = dir(fullfile(folderCat,'*.jpg'));

for i=1:length(files)
    disp(i);
    filename = files(i,1).name;
    img = imread([folderCat filename]);
    img_gray=rgb2gray(img);
    im=imresize(img_gray,[128 128]);
    J = integralImage(im);
    Sum_w=J(26,28)-J(26,25)-J(24,28)+J(24,25);
    Sum_b=J(49,49)-Sum_w;
    haar=Sum_w-Sum_b;
    M(725+i,:)=haar;
    disp(haar);

end


xlswrite('Assignment2\dataCat.xlsx',M);







  



























