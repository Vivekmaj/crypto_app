import streamlit as st
import streamlit.components.v1 as stc
from fcast_app import run_fcast_app
from binance_app import run_binance



html_temp = """
		<div style="background-color:tomato;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Stock Forecast & Binance Price App </h1>
		</div>
		"""

def main():

    stc.html(html_temp)
    

    menu = ['Home', 'Forecast Section', 'Binance Prices']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write("""
        ### App Content
            - Forecast Section: Forecast Stocks using 5,000 MC Simulations.
            - Binance Prices: View price data for crypto pairs from Binance API.
        """)

        st.markdown('### Note')
        st.write('NOTE: Due to the extreme volatile nature of certain crypto currencies and a lack of sufficient data, the MC model may not be the best approach to use.')
        
        st.markdown('### Disclaimer')
        st.write('The material on this app is purely for educational purposes and should not be taken as professional investment advice. Invest at your own discretion.')
    
    elif choice == 'Forecast Section':
        run_fcast_app()

    elif choice == 'Binance Prices':
        run_binance()

if __name__ == '__main__':
    main()