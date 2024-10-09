
% main.m
% Main script to execute the full workflow of preprocessing, generating scalograms,
% training, and evaluating the SCNN model for sleep apnea detection.

% Load and preprocess the data
raw_data = load('data/ecg_data.mat'); % Replace with actual data file path
processed_data = data_preprocessing(raw_data);

% Split the dataset
[train_data, val_data, test_data] = split_dataset(processed_data, 0.7, 0.15);

% Generate scalograms for each set
Fs = 40.9;  % Sampling frequency
train_scalograms = generate_scalograms(train_data, Fs);
val_scalograms = generate_scalograms(val_data, Fs);
test_scalograms = generate_scalograms(test_data, Fs);

% Define the CNN model layers
layers = scnn_model();

% Train the model
train_labels = categorical(load('data/train_labels.mat')); % Replace with actual labels file
val_labels = categorical(load('data/val_labels.mat')); % Replace with actual labels file
trained_model = train_model(train_scalograms, train_labels, val_scalograms, val_labels, layers);

% Evaluate the model
test_labels = categorical(load('data/test_labels.mat')); % Replace with actual labels file
metrics = evaluate_model(trained_model, test_scalograms, test_labels);

% Display metrics
disp('Model Performance:');
disp(metrics);
