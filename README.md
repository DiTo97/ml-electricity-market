# Machine Learning Methods for the Electricity Market

**Authors:** Federico Minutoli and Gianvito Losapio

We present a pair-project in which we tackle the same problem proposed by the IREN company during the C1A0 hackathon held on 14-15/11/2019. We tried to mix project of types 1 and 2 with the aim of considering a full real-case scenario concerning the electricity market. As for the type 2 part of the project, we provide a detailed data processing and a simple time-series analysis in the notebook file *code/preprocessing/Time-series analysis.ipynb*. As for the type 1 part of the project, a detailed formulation of two different algorithms follows: Random forests (by Gianvito Losapio) and Gradient boosting (by Federico Minutoli). Both of the algorithms have been implemented from scratch. The structure of the code is presented in the notebook file code/Problem.ipynb as well as its use to solve the main problem. Results and comparisons of the methods are provided.


Mix project of types 1 and 2: same problem proposed by the IREN company during C1A0 Hackathon.

The Day-Ahead Market (MGP=MercatoGiornoPrima), is the market for the trading of electricity supply offers and demand bids for each hour of the next day in Italy.All electricity operators may participate in the MGP.

Participants submit offers (bids) where they specify the quantity and the minimum (maximum) price they are willing to sell (purchase).The accepted offers are those with a submitted price not larger than the marginal clearing price (obtained by Gestore Mercati Energetici from the intersection of the demand and supply curves). The accepted bids are those with a submitted price not lower than the PUN (Prezzo Unico Nazionale – national single price, agreed daily)


<br>

[Live Notebook](https://colab.research.google.com/drive/1VTr14Lc-uP8Ss2yH9bPq1lknbLg1tH3u?usp=sharing) on Colab

<br>

