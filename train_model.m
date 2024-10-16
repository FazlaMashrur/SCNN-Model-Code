
% train_model.m
% This function trains the SCNN model using the provided training data and labels.

function trained_model = train_model(train_data, train_labels, val_data, val_labels, layers)
    % Specify training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.0004, ...
        'MaxEpochs', 35, ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {val_data, val_labels}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu', ...
        'L2Regularization', 0.0001, ...
        'Momentum', 0.90);

    % Train the model
    trained_model = trainNetwork(train_data, train_labels, layers, options);
end
