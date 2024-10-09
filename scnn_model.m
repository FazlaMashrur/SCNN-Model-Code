
% scnn_model.m
% This function defines the architecture of the SCNN model for sleep apnea detection.

function layers = scnn_model()
    % Define the CNN layers
    layers = [
        imageInputLayer([32 32 3], 'Name', 'input')

        convolution2dLayer(3, 512, 'Stride', 1, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        softplusLayer('Name', 'softplus1')
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'maxpool1')

        convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        softplusLayer('Name', 'softplus2')
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'maxpool2')

        convolution2dLayer(7, 128, 'Stride', 1, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        softplusLayer('Name', 'softplus3')
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'maxpool3')

        fullyConnectedLayer(2, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
end
