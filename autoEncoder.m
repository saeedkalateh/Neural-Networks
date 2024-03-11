%% clear
clear
clc
%% load data
[x1,x1_label] = loadMNIST(0,[1]);
[x2,x2_label] = loadMNIST(0,[2]);
[x3,x3_label] = loadMNIST(0,[3]);
[x4,x4_label] = loadMNIST(0,[4]);
[x5,x5_label] = loadMNIST(0,[5]);
[x6,x6_label] = loadMNIST(0,[6]);
[x7,x7_label] = loadMNIST(0,[7]);
[x8,x8_label] = loadMNIST(0,[8]);
[x9,x9_label] = loadMNIST(0,[9]);
[x10,x10_label] = loadMNIST(0,[10]);

%% Create a training set with only 2 classes
train_data=vertcat(x4,x6);
train_data=train_data';

labels=vertcat(x1_label,x8_label);

%% Train an autoencoder on the new, reduced training set 
myAutoencoder = trainAutoencoder(train_data, 100, 'MaxEpochs', 300);
myEncodedData = encode(myAutoencoder,train_data);

%% Encode the different classes using the encoder obtained
myEncodedX1 = encode(myAutoencoder,x1');
myEncodedX2 = encode(myAutoencoder,x2');
myEncodedX3 = encode(myAutoencoder,x3');
myEncodedX4 = encode(myAutoencoder,x4');
myEncodedX5 = encode(myAutoencoder,x5');
myEncodedX6 = encode(myAutoencoder,x6');
myEncodedX7 = encode(myAutoencoder,x7');
myEncodedX8 = encode(myAutoencoder,x8');
myEncodedX9 = encode(myAutoencoder,x9');
myEncodedX10 = encode(myAutoencoder,x10');

%% Plot the data using the "plotcl" function
plotcl(myEncodedX1(:,[1 8])',x1_label)
figure()
plotcl(myEncodedX5(:,[1 8])',x5_label)
%% Predict
figure()
x1_pred=predict(myAutoencoder,x1(1,:)');
subplot(5,4,1);
imshow(reshape(x1(1,:)', [28 28]));
title("Input");
subplot(5,4,2);
imshow(reshape(x1_pred, [28 28]));
title("Predicted");

x2_pred=predict(myAutoencoder,x2(1,:)');
subplot(5,4,3);
imshow(reshape(x2(1,:)', [28 28]));
title("Input");
subplot(5,4,4);
imshow(reshape(x2_pred, [28 28]));
title("Predicted");

x3_pred=predict(myAutoencoder,x3(1,:)');
subplot(5,4,5);
imshow(reshape(x3(1,:)', [28 28]));
title("Input");
subplot(5,4,6);
imshow(reshape(x3_pred, [28 28]));
title("Predicted");

x4_pred=predict(myAutoencoder,x4(1,:)');
subplot(5,4,7);
imshow(reshape(x4(1,:)', [28 28]));
title("Input");
subplot(5,4,8);
imshow(reshape(x4_pred, [28 28]));
title("Predicted");

x5_pred=predict(myAutoencoder,x5(1,:)');
subplot(5,4,9);
imshow(reshape(x5(1,:)', [28 28]));
title("Input");
subplot(5,4,10);
imshow(reshape(x5_pred, [28 28]));
title("Predicted");

x6_pred=predict(myAutoencoder,x6(1,:)');
subplot(5,4,11);
imshow(reshape(x6(1,:)', [28 28]));
title("Input");
subplot(5,4,12);
imshow(reshape(x6_pred, [28 28]));
title("Predicted");

x7_pred=predict(myAutoencoder,x7(1,:)');
subplot(5,4,13);
imshow(reshape(x7(1,:)', [28 28]));
title("Input");
subplot(5,4,14);
imshow(reshape(x7_pred, [28 28]));
title("Predicted");

x8_pred=predict(myAutoencoder,x8(1,:)');
subplot(5,4,15);
imshow(reshape(x8(1,:)', [28 28]));
title("Input");
subplot(5,4,16);
imshow(reshape(x8_pred, [28 28]));
title("Predicted");

x9_pred=predict(myAutoencoder,x9(1,:)');
subplot(5,4,17);
imshow(reshape(x9(1,:)', [28 28]));
title("Input");
subplot(5,4,18);
imshow(reshape(x9_pred, [28 28]));
title("Predicted");

x10_pred=predict(myAutoencoder,x10(1,:)');
subplot(5,4,19);
imshow(reshape(x10(1,:)', [28 28]));
title("Input");
subplot(5,4,20);
imshow(reshape(x10_pred, [28 28]));
title("Predicted");