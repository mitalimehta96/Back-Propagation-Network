% AIM- BACK PROPAGATION NETWORK IMPLEMENTATION WITH BINARY SIGMOIDAL FUNCTION FOR 1 OUTPUT WITH 2 INPUTS AND A HIDDEN LAYER
% BINARY SIGMOIDAL -> F(X) = 1/1+e^(-X)
 
clc;
clear all;
 
x = [0 1]; % input vector
t = 1; % target value. If there are more than 1 output neurons, there will be more than 1 targets
alpha = 0.5; % Learning rate
counter = 1; % to count the number of epochs
v11 = input('Weight v11= ');
v21 = input('Weight v21= ');
v01 = input('Weight v01= ');
v12 = input('Weight v12= ');
v22 = input('Weight v22= ');
v02 = input('Weight v02= ');
w1 = input('Weight w1= ');
w2 = input('Weight w2= ');
w0 = input('Weight w0= ');

% you can use these weight values for testing the code if you don't want to input
% values again and again
% v11 = 0.6; v01 = 0.3; v12 = -0.3; v22 = 0.4; v02 = 0.5; w1 = 0.4; 
% v21 = -0.1; w2 = 0.1; w0 = -0.2;
 
zin1 = v11*x(1) + v21*x(2) + v01;
zin2 = v12*x(1) + v22*x(2) + v02;
 
z1 = 1/(1+exp(-zin1));
z2 = 1/(1+exp(-zin2));
 
yin = w1*z1 + w2*z2 + w0;
 
y = 1/(1+exp(-yin));
 
while( y ~= t && counter <= 1000)
    %Error calculation between output and hidden layer
    del1 = (t-y)*y*(1-y);
    dw1 = alpha * del1 * z1;
    dw2 = alpha * del1 * z2;
    dw0 = alpha * del1;
    %Update weights between output and hidden layer
    w1 = w1 + dw1;
    w2 = w2 + dw2;
    w0 = w0 + dw0;
    %Error calculation between input and hidden layer
    delin1 = del1 * w1;
    delin2 = del1 * w2;
    dell1 = delin1 * z1 * (1-z1);
    dell2 = delin2 * z2 * (1-z2);
    %Update weights between input and hidden layer
    v11 = v11 + alpha * dell1 * x(1);
    v21 = v21 + alpha * dell1 * x(2);
    v01 = v01 + alpha * dell1;
    v12 = v12 + alpha * dell2 * x(1);
    v22 = v22 + alpha * dell2 * x(2);
    v02 = v02 + alpha * dell2;
    
    zin1 = v11*x(1) + v21*x(2) + v01;
    zin2 = v12*x(1) + v22*x(2) + v02;
 
    z1 = 1/(1+exp(-zin1));
    z2 = 1/(1+exp(-zin2));
 
    yin = w1*z1 + w2*z2 + w0;
 
    y = 1/(1+exp(-yin));
    counter = counter + 1 ;
    if(mod(counter,100) == 0)
        fprintf('\nValue after EPOCH %d is:',counter);
        fprintf('\nv11 = %3.5f', v11);
        fprintf('\nv21 = %3.5f', v21);
        fprintf('\nv01 = %3.5f', v01);
        fprintf('\nv12 = %3.5f', v12);
        fprintf('\nv22 = %3.5f', v22);
        fprintf('\nv02 = %3.5f', v02);
        fprintf('\nw1 = %3.5f', w1);
        fprintf('\nw2 = %3.5f', w2);
        fprintf('\nw0 = %3.5f\n', w0);
    end
end
 
