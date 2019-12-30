# capstone
![Image description](link-to-image)
# Abstract
The buildings sector accounts for about 76%* of electricity use and 40% of all U.S. primary energy use and associated greenhouse gas (GHG) emissions. Reducing the energy consumption in buildings in order to meet national energy and environmental challenges and to reduce costs to building owners and tenants become crucial.

There is no best way to motivate people to cut the energy waste in their homes and businesses? But, there are energy efficiency programs that offer rebates and incentives upfront to promote smarter energy use through things like highly efficient appliances and weatherization technologies.

This is my capstone project on great energy predictor. I built a special kind of Recurrent Neural Networks (RNN) called "Long Short Term Memory" networks -usually just called "LSTMs", which is capable of learning long-term dependencies in time series analysis. I was able to predict the baseline energy consumption - for case study office building - at 14.7% Mean Squared Precent Error.
I trained my model on google cloud machine with 8xCPU and 30GB RAM using the Keras API with the TensorFlow backend.


# Motivation
One among many energy efficiency programs is "Pay-for-performance", which is commonly abbreviated as P4P. P4P program track and rewards end side energy use reduction as they occur as opposed to the more common approach of estimating reduction in advance of installation and offering upfront rebates in a lump-sum payment. This is the motive for this capstone project to predict baseline energy consumption.

# Dataset
I got the Dataset from kaggle commpetation organized by ASHERE. The Dataset contains 1449 buildings located worldwide with differnt operation

# Preprocessing
LSTM model requires a three-dimentional input and
