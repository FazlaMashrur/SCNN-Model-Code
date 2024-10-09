
% evaluate_model.m
% This function evaluates the trained model on the test data and computes performance metrics.

function metrics = evaluate_model(trained_model, test_data, test_labels)
    % Predict labels for test data
    predicted_labels = classify(trained_model, test_data);
    
    % Calculate metrics
    accuracy = sum(predicted_labels == test_labels) / numel(test_labels) * 100;
    [confusion_mat, ~] = confusionmat(test_labels, predicted_labels);
    sensitivity = confusion_mat(2,2) / (confusion_mat(2,2) + confusion_mat(2,1)) * 100;
    specificity = confusion_mat(1,1) / (confusion_mat(1,1) + confusion_mat(1,2)) * 100;
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity);

    % Store results in a struct
    metrics = struct('Accuracy', accuracy, 'Sensitivity', sensitivity, ...
                     'Specificity', specificity, 'F1_Score', f1_score);
end
