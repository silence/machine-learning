
%% Clear and Close Figures
clear ;  clc;

fprintf('Loading data ...\n');

%% Load Data
data = load('data.txt');
X = data(1:820, 1:12);
y = data(1:820, 13);
m = length(y);

fprintf('Program paused. Press enter to continue.\n');


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.001;
num_iters = 30000;

% Init Theta and Run Gradient Descent 
theta = zeros(13, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
hold on;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
%fprintf(' %f \n', theta);
fprintf('\n');


%verify_data = data(701:827,:);
%predict_result = theta' * [ones(127,1),(verify_data(:,1:12)-mu)./sigma]';
%true_data = data(701:827,13);
verify_data = data(1:820,:);
predict_result = theta' * [ones(820,1),(verify_data(:,1:12)-mu)./sigma]';
true_data = data(1:820,13);
n=abs(predict_result' - true_data);

for i = 1:820
    n(i) = n(i)/true_data(i);
end

fprintf('预测结果准确率为%f %% \n',(1-sum(n)/820)*100);



%% Clear and Close Figures

%% Load Data
data = load('data的副本.txt');
X = data(1:820, 1:12);
y = data(1:820, 13);
m = length(y);

fprintf('Program paused. Press enter to continue.\n');


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.001;
num_iters = 30000;

% Init Theta and Run Gradient Descent 
theta = zeros(13, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph

plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold off;

legend('标准对应','斜着对应');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
%fprintf(' %f \n', theta);
fprintf('\n');


%verify_data = data(701:827,:);
%predict_result = theta' * [ones(127,1),(verify_data(:,1:12)-mu)./sigma]';
%true_data = data(701:827,13);
verify_data = data(1:820,:);
predict_result = theta' * [ones(820,1),(verify_data(:,1:12)-mu)./sigma]';
true_data = data(1:820,13);
n=abs(predict_result' - true_data);

for i = 1:820
    n(i) = n(i)/true_data(i);
end

fprintf('预测结果准确率为%f %% \n',(1-sum(n)/820)*100);

%% normal equation
data = load('data.txt');
X = data(1:820, 1:12);
y = data(1:820, 13);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

verify_data = data(1:820,:);
predict_result = theta' * [ones(820,1),(verify_data(:,1:12))]';
true_data = data(1:820,13);
n=abs(predict_result' - true_data);

for i = 1:820
    n(i) = n(i)/true_data(i);
end

fprintf('标准方程预测结果准确率为%f %% \n',(1-sum(n)/820)*100);


%% data的副本

data = load('data的副本.txt');
X = data(1:820, 1:12);
y = data(1:820, 13);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

verify_data = data(1:820,:);
predict_result = theta' * [ones(820,1),(verify_data(:,1:12))]';
true_data = data(1:820,13);
n=abs(predict_result' - true_data);

for i = 1:820
    n(i) = n(i)/true_data(i);
end

fprintf('标准方程预测结果准确率为%f %% \n',(1-sum(n)/820)*100);









