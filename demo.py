import streamlit as st
import pickle
import pandas as pd


teams = ['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bengaluru',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants']

cities = ['Bangalore', 'Chandigarh', 'Kolkata', 'Chennai', 'Mumbai',
       'Hyderabad', 'Cape Town', 'Port Elizabeth', 'Durban',
       'East London', 'Johannesburg', 'Centurion', 'Kimberley', 'Cuttack',
       'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Delhi', 'Abu Dhabi',
       'Ranchi', 'Pune', 'Indore', 'Bengaluru', 'Ahmedabad', 'Dubai',
       'Sharjah', 'Navi Mumbai', 'Lucknow', 'Mohali']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL WIN PREDICTOR')

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams)) 

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120- (overs*6)
    remaining_wickets = 10 - wickets
    current_run_rate = score/overs
    required_run_rate = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                  'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'remaining_wickets':[remaining_wickets],
                  'total_runs_x':[target],'current_run_rate':[current_run_rate],'required_run_rate':[required_run_rate]})
    
    st.table(input_df)

    result = pipe.predict_proba(input_df)
    st.text(result)

    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "-" + str(round(win*100)) + "%")
    st.header(bowling_team + "-" + str(round(loss*100)) + "%")




    

