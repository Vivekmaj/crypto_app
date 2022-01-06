import streamlit as st
from datetime import date
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
matplotlib.use('agg')

START = "2008-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # data.reset_index(inplace=True)
    return data

def run_fcast_app():

    st.header('**Monte Carlo Simulations**')

    default = 'AAPL'
    user_input = st.text_input("Insert Ticker Symbol", default)

    period = st.slider('Forecast Duration (Days):', 7, 60)

    if st.button('Forecast!'):

        data = load_data(user_input)

        data = data[['Close']]

        y=np.log10(data["Close"])
        data['log'] = y
        data = data.drop('Close', axis = 1)

        data['log_vals'] = data.pct_change()

        ## In Sample testing
        train = data[:data.shape[0] - period]
        test = data.tail(period)

        change = train['log_vals']
        last_val = train['log'][-1]

        ## No of Simulations
        num_simulations = 5000
        num_days = len(test)
        simulations_df = pd.DataFrame()

        for x in range(num_simulations):
            count = 0
            daily_vol = change.std()
            
            series = []
            
            val = last_val * (1 + np.random.normal(0, daily_vol))
            series.append(val)
            
            for y in range(num_days):
                if count == num_days - 1:
                    break
                value = series[count] * (1 + np.random.normal(0, daily_vol))
                series.append(value)
                count += 1
                
            simulations_df[x] = series

        cols = list(simulations_df.T.columns)
        ninety = []
        seven_five = []
        twenty_five = []
        ten =  []

        for i in range(len(cols)):
            ninety.append(np.percentile(list(simulations_df.T[cols[i]]), 90))
            seven_five.append(np.percentile(list(simulations_df.T[cols[i]]), 75))
            ten.append(np.percentile(list(simulations_df.T[cols[i]]), 10))
            twenty_five.append(np.percentile(list(simulations_df.T[cols[i]]), 25))

        av = pd.DataFrame(simulations_df.median(axis = 1), columns = ['Average'])
        sf = pd.DataFrame(seven_five, columns = ['P75'])
        tf = pd.DataFrame(twenty_five, columns = ['P25'])
        nt = pd.DataFrame(ninety, columns = ['P90'])
        tn = pd.DataFrame(ten, columns = ['P10'])

        pred_df = pd.concat([av,sf,tf,nt,tn], axis = 1)

        with st.expander('View Testing Results'):

            fig = plt.figure(figsize = (10,5))

            plt.plot(pred_df['P90'], ls = '-', lw = 0.1)
            plt.plot(pred_df['P75'], ls = '-', lw = 0.1)
            plt.plot(pred_df['P25'], ls = '-', lw = 0.1)
            plt.plot(pred_df['P10'], ls = '-', lw = 0.1)

            plt.plot(pred_df['Average'], label = 'Average Pred', marker = 'o', c = 'black')
            plt.plot(list(test['log']), label = 'Test Data', marker = 'o', c = 'green')

            plt.fill_between(pred_df.index, pred_df['Average'], pred_df['P75'], color = 'yellow', alpha = 1.0)
            plt.fill_between(pred_df.index, pred_df['Average'], pred_df['P25'], color = 'yellow', alpha = 1.0)

            plt.fill_between(pred_df.index, pred_df['Average'], pred_df['P90'], color = 'orange', alpha = 0.2)
            plt.fill_between(pred_df.index, pred_df['Average'], pred_df['P10'], color = 'orange', alpha = 0.2)


            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            plt.grid(alpha = 0.2, c = 'r')

            plt.xlabel('Forecast Timeline')
            plt.ylabel(data.columns[0])
            plt.title('Insample Test')

            st.pyplot(fig)

        ## Out-of-Sample Forecast
        change = data['log_vals']
        last_val = data['log'][-1]

        ## No of Simulations
        num_simulations = 5000
        num_days = period
        simulations_df = pd.DataFrame()

        for x in range(num_simulations):
            count = 0
            daily_vol = change.std()
            
            series = []
            
            val = last_val * (1 + np.random.normal(0, daily_vol))
            series.append(val)
            
            for y in range(num_days):
                if count == num_days - 1:
                    break
                value = series[count] * (1 + np.random.normal(0, daily_vol))
                series.append(value)
                count += 1
        
            simulations_df[x] = series

        with st.expander('View Potential Price Paths'):
            fig = plt.figure(figsize = (10,5))
            plt.plot(simulations_df)
            plt.grid(alpha = 0.2, c = 'r')
            plt.xlabel('Forecast Timeline')
            plt.ylabel(data.columns[0])
            plt.title('Monte Carlo Simulations')
            st.pyplot(fig)

        cols = list(simulations_df.T.columns)
        ninety = []
        seven_five = []
        twenty_five = []
        ten =  []

        for i in range(len(cols)):
            ninety.append(np.percentile(list(simulations_df.T[cols[i]]), 90))
            seven_five.append(np.percentile(list(simulations_df.T[cols[i]]), 75))
            ten.append(np.percentile(list(simulations_df.T[cols[i]]), 10))
            twenty_five.append(np.percentile(list(simulations_df.T[cols[i]]), 25))

        future_dates = pd.date_range(start = data.index[-1], periods = num_days+1, freq = 'D').to_pydatetime().tolist()
        del future_dates[0]

        av = pd.DataFrame(simulations_df.median(axis = 1), columns = ['Average'])
        fut = pd.DataFrame(future_dates, columns = ['Date'])
        sf = pd.DataFrame(seven_five, columns = ['P75'])
        tf = pd.DataFrame(twenty_five, columns = ['P25'])
        nt = pd.DataFrame(ninety, columns = ['P90'])
        tn = pd.DataFrame(ten, columns = ['P10'])

        pred_df = pd.concat([fut,av,sf,tf,nt,tn], axis = 1)

        pred_df = pred_df.set_index('Date')
        cols = pred_df.columns
        pred_df_final = 10 ** pred_df[cols]
        pred_df_final[cols] = pred_df_final[cols].astype('int64')

        with st.expander('View Forecast Results Dataframe'):
            st.write(pred_df_final)

        boeing_final = 10 ** data['log']
        bng = pd.DataFrame(boeing_final)
        bng.columns = ['Close']

        with st.expander(f'View Forecast Plot for {period} Days'):
            fig = plt.figure(figsize = (10,5))

            plt.plot(pred_df_final['P90'], ls = '-', lw = 0.1)
            plt.plot(pred_df_final['P75'], ls = '-', lw = 0.1)
            plt.plot(pred_df_final['P25'], ls = '-', lw = 0.1)
            plt.plot(pred_df_final['P10'], ls = '-', lw = 0.1)

            plt.plot(pred_df_final['Average'], label = 'Avg Forecast Value', c = 'black', ls = '--', markersize = 5)
            plt.plot(bng['Close'].tail(60), label = 'Last 60 Days Close', c = 'blue')

            plt.fill_between(pred_df_final.index, pred_df_final['Average'], pred_df_final['P75'], color = 'yellow', alpha = 1.0)
            plt.fill_between(pred_df_final.index, pred_df_final['Average'], pred_df_final['P25'], color = 'yellow', alpha = 1.0)

            plt.fill_between(pred_df_final.index, pred_df_final['Average'], pred_df_final['P90'], color = 'orange', alpha = 0.2)
            plt.fill_between(pred_df_final.index, pred_df_final['Average'], pred_df_final['P10'], color = 'orange', alpha = 0.2)

            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            plt.grid(alpha = 0.2, c = 'r')

            plt.xlabel('Forecast Timeline')
            plt.ylabel(bng.columns[0])

            plt.xticks(rotation = 90)

            plt.title(user_input + ' '+ 'Forecast')

            st.pyplot(fig)