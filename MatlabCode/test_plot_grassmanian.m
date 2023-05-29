
data = load('../PythonCode/mnist_experiment/predictions_dim0_qr.csv');

markers = {'o', '+', 'diamond', '.'};
allMethods = {'Flag Median', 'Flag Mean', 'Real Flag Median', 'Real Flag Mean'};
numMethods = length(allMethods);

x = data(:,1);
y = data(:,2);

n = length(x);
x = reshape(x, numMethods, fix(n/numMethods));
y = reshape(y, numMethods, fix(n/numMethods));

ms = 60;
f = figure;
beautify_plot; 
box on;

for i=1:numMethods
    hold on; plot(x(i,:), y(i,:),'Marker',markers{i}, 'LineWidth', 2, 'DisplayName', allMethods{i});
end
legend([allMethods(1:numMethods)], 'Location', 'northwest');
xlabel('Added Nines');
ylabel('Prediction');
% axis([0 0.82 0 60]);
