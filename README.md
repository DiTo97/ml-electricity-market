# Machine Learning Methods for the Electricity Market

**Authors:** Federico Minutoli and Gianvito Losapio

We present a pair-project in which we tackle the same problem proposed by the [IREN](https://www.gruppoiren.it/home) company during the C1A0 hackathon held on 14-15/11/2019 in Genoa. We tried to mix project of types 1 and 2 with the aim of considering a full real-case scenario concerning the electricity market. As for the type 2 part of the project, we provide a detailed data processing and a simple time-series analysis in the notebook file *code/preprocessing/Time-series analysis.ipynb*. As for the type 1 part of the project, a detailed formulation of two different algorithms follows: Random forests (by Gianvito Losapio) and Gradient boosting (by Federico Minutoli). Both of the algorithms have been implemented from scratch. The structure of the code is presented in the notebook file code/Problem.ipynb as well as its use to solve the main problem as presented by IREN. Results and comparisons of the methods are provided.

During the C1A0 hackathon we were asked to design an AI model able to predict the day-ahead market (MGP) for the subsequent lag of 14 days. At the time, we used the LightGBM algorithm as its speed was very well suited for a competitive setting. This project, instead, focuses more on gaining a deeper understanding of its capabilities.

## MGP

The Day-ahead Market (or Mercato del Giorno Prima, MGP, in Italian), is the market designed for trading electricity supply offers and demand bids for each hour of the next day in Italy. All major electricity operators may participate in the MGP by submitting offers (bids) where they specify the quantity and the minimum (maximum) price they are willing to sell (purchase). The accepted offers are those with a submitted price not larger than the marginal clearing price (MGP), obtained from the intersection of the demand and supply curves. The accepted bids are similarly those with a submitted price not lower than the daily-agreed National Single Price (or Prezzo Unico Nazionale, PUN).

All of these factors greatly influence the understanding of the MGP chain.

## Folder structure

- `code` folder contains the Random forest and the Gradient boosting implementations from scratch as well as a notebook to play with, _Problem.ipynb_, with all the key experiments that we ran on the MGP data culminating in the comparison with LightGBM in terms of RMSE of the MGP values.
