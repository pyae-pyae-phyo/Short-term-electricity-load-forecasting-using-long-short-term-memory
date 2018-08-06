# Self_Introduction

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/self_intro_1.JPG?raw=true)


# Project Title: Short-term-electricity-load-forecasting-using-long-short-term-memory

## 1. Problem Statement

Technological advancement necessitates effective operation in generation, transmission and distribution of electricity in electric companies and industries. The amount of electricity generation and electricity consumption should be balanced because of no developed system to store generated electricity or to produce electrical energy at a sudden supply demand. Imbalanced demands of electricity generation cause economic losses and dissatisfaction of users. Thus, it is important for the electricity providers to maintain the balance among electricity generation, transmission, distribution and consumer services. Electricity load forecasting is one of the necessary techniques to improve the balance of electricity, but there are difficulties in coping with many complex factors such as weather, temperature, seasonal variation, etc., which are all non-linear. Recent studies tried to solve these issues by employing various machine learning methods, and deep learning techniques have been paid more attention. Henrique et. al [1] reviewed and evaluated neural networks for short-term load forecasting focusing on Artificial Neural Network (ANN). However, the presence of back propagation and complex multiple layers for simple compute gradients reduced the efficiency of the solution.

Many recent studies have been conducted on load forecasting using different deep learning techniques. Some reviewed papers are shown in Table 1. Long short-term memory (LSTM) based recurrent neural network (RNN) is one of popular methods for load forecasting. This model has been used on different input variables and datasets from various countries. Most of the studies mention that the selection of input variables is one of important things to consider for load forecasting to achieve better performance. Thus, this project specifically focuses on an efficient input structure using LSTM model.

Table 1.  A review of Deep Leaning (DL) models used in load forecasting papers

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/related_works.JPG?raw=true)

The data for this project is provided by Electricity Generating Authority of Thailand (EGAT). In this project, the input variables of the proposed model are considered as load of previous week same day (Lt(d-7)), load of yesterday (two periods before) (Lt-2(d-1)), load of yesterday (one period before) (Lt-1(d-1)), load of yesterday at the same period (Lt(d-1)), temperature of yesterday at the same period (Tt(d-1)), monthly seasonal index (SI) which is monthly load divided by yearly load, and January to December as 12 inputs which have values 0 and 1 (e.g., if it is January -> only Jan is 1, others are 0) to forecast the load (Ft(d)). This project ensures the appropriate selection of parameters during the training process which affects the forecasting performance.

## 2. Goal setting

### 2.1 Long-term goals (to show where this project is located in my graduate research)

- To develop an efficient forecasting system for electricity generation companies and industries t for reducing the generating and operating cost.
- To investigate the application of appropriate techniques and tools of forecasting on electricity for Thailand with minimum forecasting error.
- To investigate the application of the generated results as a guideline for the better performance of different electric industries in other Asian countries.

### 2.2 Short-term goals (as the goals for this camp)

- To develop an appropriate model of LSTM-RNN for electricity balance forecasts.
- To evaluate my model with other related studies.

## 3. Proposed framework/methodology

### 3.1 Data Collection and Preprocessing

The historical load and temperature data (from 2009 to 2017) are provided by Electricity Generating Authority of Thailand (EGAT). This data has been collected from five different regions: Central, Bangkok and Metropolitan, South, North and North-East in Thailand. There is a low electricity demand with only one peak load curve at night in the three regions: South, North, and North-East. On the other hand, there is a high electricity demand with three peak load curves in two regions: Bangkok and Metropolitan, and Central.For the load data, this project will use net-peak load which is recorded every 30 minutes for the whole country. In the temperature file, not only temperature but also other factors such as dewpoint, relative humidity, pressure, wind, rainfall, and cloud are included. The temperature readings are recorded every 30 minutes for the whole country. In other words, all load and temperature data have 48 periods per day. In this project, the proposed model uses the data from 2016 as the training set and data from 2017 as the testing set to forecast the daily load demand. This time-sequential data need preprocessing particularly because there are many missing values, and outliers in the original raw data. If the outliers are included in the training data, the accuracy performance of load predictions might be lower. In this project, a local regression filtering technique with the usage of regression analysis is applied in order to filter and smoothen the raw data. The filtering technique fits a local regression function to the data within a chosen neighbourhood of data points. A chosen neighbourhood specified by percentage of data points is known as a smoothing parameter (0 < smooth <= 1). The larger the smoothing parameter, the smoother the graphed function. For calculating smoothed values, this filtering technique will specify weight for every data point in the selected window by using regression weight function.

### 3.2 Methodology

The goal of this project is to develop a forecasting system of electricity balance between supply and demand using the LSTM based RNN technique and evaluate the method by comparing it to other techniques. The overview of the process in the system is shown in Figure 1. Firstly, the system is loaded with a dataset consisting eighteen input variables. The data is normalized by using the min-max scaling method. The data is then splitted into the training and testing sets. Next, a LSTM network is constructed and trained. After that, the trained LSTM network is used for prediction.

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/lstm.JPG?raw=true)

Figure 1. A process of electricity load forecasting

Apowerful type of neural network designed to handle sequence dependence is the recurrent neural network. In this project, the LSTM network is a recurrent neural network that is trained using backpropagation through time to overcome the vanishing gradient problem. First of all, the eighteen input variables are imported into the LSTM network. In the LSTM network, the input units are fully connected to a hidden layer. The cell outputs are fully connected to the cell inputs, to all gates, and to the output units. During the model fitting, mean squared error is used as loss function for optimizing the parameters of the LSTM model and Adam is used as the optimization algorithm for the loss function.
The electric load data is considered as time-series data because it is recorded every 30 minutes. In time-series data, the data collected at a time t naturally relates to the data at the previous adjacent time (i.e., at the time t – 1). A LSTM network deals with this characteristic by using hidden-state information obtained during the network training with the data at the time t – 1 to train the network with the data at the time t. Figure 2, for example, illustrates how the LSTM network works, where xt-1, xt, and xt+1 are the input at times t - 1, t, and t + 1, respectively, s0, s1, and s2 are hidden-state information of hidden layers, and ot-1, ot, and ot+1 are the output at times t - 1, t, and t + 1, respectively.

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/lstm_1.JPG?raw=true)

Figure 2. The working process of LSTM networks

## 4. Result and Discussion

Table 2. Average monthly MAPE in 2017

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/mape.JPG?raw=true)

The aim of this project is to minimize mean absolute percentage error (MAPE) which is used as accuracy measurement. The average monthly MAPE in 2017 is shown in Table 2. According to the table, error in December are highest because of long holidays, tourism and public events. As the consequence of results in December, January is getting higher MAPE. Moreover, the highest temperature is affected on the accuracies in March, April and May for both LSTM and DBN models. Overall, the LSTM model outperforms the DBN model in every months. Therefore, LSTM model can predict nearly the minimum MAPE (0.8852) and the maximum MAPE (21.3393) which are shown in Figure 3 and Figure 4.

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/min_mape.JPG?raw=true)

Figure 3. Minimum MAPE in 2017

![](https://github.com/pyae-pyae-phyo/Short-term-electricity-load-forecasting-using-long-short-term-memory/blob/master/picture/max_mape.JPG?raw=true)

Figure 4. Maximum MAPE in 2017

## Application & Computation Time

- Python 3.6.4
- Anaconda Navigator
- Computation time - 30min for each dataset

## Future Work

- Try on the new input structure more. 
- Train the model using more datasets (up to 5 years)
- Use consumption energy (MWh) instead of net peak (MW)
- Implement the model to reduce error in December
- Apply another deep learning technique

## References

1. Hippert, H. S., Pedreira, C. E., & Souza, R. C. (2001). Neural networks for short-term load forecasting: A review and evaluation. IEEE Transactions on power systems, 16(1), 44-55.
2. Narayan, A., & Hipel, K. W. (2017, October). Long short term memory networks for short-term electric load forecasting. In Systems, Man, and Cybernetics (SMC), 2017 IEEE International Conference on (pp. 2573-2578). IEEE.
3. Zhang, B., Wu, J. L., & Chang, P. C. (2018). A multiple time series-based recurrent neural network for short-term load forecasting. Soft Computing, 22(12), 4099-4112.
4. Liu, C., Jin, Z., Gu, J., & Qiu, C. (2017, September). Short-term load forecasting using a long short-term memory network. In Innovative Smart Grid Technologies Conference Europe (ISGT-Europe), 2017 IEEE PES (pp. 1-6). IEEE.
5. Hamedmoghadam, H., Joorabloo, N., & Jalili, M. (2018). Australia's long-term electricity demand forecasting using deep neural networks.

## Acknowledgements

This project is partially supported by Jeju National University and JDC company. Data used in this project is provided by EGAT, Thailand. Therefore, I acknowledge their support for completing this project. Additionally, I would like to thank Prof.Kiyota Hashimoto for choosing me as a mentee and giving guidlines for this camp. Specifically, this MLC Jeju 2018 is meaningless for us without Prof. Yungcheol Byun, therefore I also thank to him for holding the host for us ever. I am really happy for having lovely roommate Smile :) (Thanyathorn Thanapattheerakul) who helped me alot and also thank to Aneeqa Ahmed and F4 (Son, Sumin, Sehoon and Lee) for helping and supporting us. Finally, I thank to all mentors and mentees from different countries and nothing is fun without them. I hope all of you guys have bright future and see you at the top.