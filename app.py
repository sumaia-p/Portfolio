import streamlit as st
import streamlit.components.v1 as components
import base64
import datetime

def road_safety():

    st.subheader('AI for Improving Road Safety in Bangladesh')
    d1 = st.date_input("", [datetime.date(2021, 6, 26), datetime.date(2021, 8, 14)])
    
    st.write("What's done in this project?")

    st.markdown("- Time-Series Forecast: CNN-LSTM, LSTM, Greykites, Kats: SARIMA & Prophet, and PyCaret TS Analysis.")
    st.markdown("- Times-Series Classification: HyperOpt Optimized QDA, MLJar: Ensemble, SkTime Multivariate Analysis, TPOT Optimized MLP Classifier, and PyCaret: QDA.")
    st.markdown("- Text Analytics: Topic Modelling (LDA).")
    st.markdown("- Folium Heatmaps with Time Lapse.")
    st.markdown("- Flourish Visualizations: Bar chart race showing top 15 days with most death caused by road accidents, complete statistical analysis dashboard.")

    st.write('- [See in detail on GitHub](https://github.com/SumaiaParveen/Omdena-AI-for-Road-Safety-In-Bangladesh)')

def tfg_aws_pred():

    st.subheader('Cloud Instance Price Prediction')

    d1 = st.date_input("", [datetime.date(2021, 3, 20), datetime.date(2021, 6, 7)])

    st.write(" ")

    st.write("What's done in this project?")

    st.markdown("- Collected virtual machine pricing data using Google Cloud Billing API and Azure Retail Price API. Wrote scripts to pull hourly and daily data and integrate with previous timestamps' datasets. Extracted useful data from JSON and converted it to tabular format using Pandas.")

    st.markdown("- Analyzed data on Google BigQuery, pulled data to Pandas from BigQuery using API. Visualized BigQuery data on Google Data Studio and JupyterLab.")

    st.markdown("- Collected data from various sources using AWS CLI, AWS API, and several Python packages i.e., boto3. Wrangled, preprocessed, analyzed, post-processed, and visualized the data on JulyterLab and Flourish.")

    st.markdown("- Various Machine Learning and Deep Learning models including SARIMAX, Prophet, MLP, etc. were employed to predict the price of VM instances. Finally, the one with the best scores was chosen for deployment.")

    st.write(" ")

    st.write("What's inside the notebooks?")
    st.write("Various Machine Learning and Deep Learning models including SARIMAX, Prophet, MLP, etc. were employed to predict the price of spot/preemptible instances. Finally, the one with the best scores was chosen for deployment.")

    st.write('Time-Series Forecast')
    st.markdown('- [CNN-LSTM](https://colab.research.google.com/drive/1Mh_VuTY_zqRtmDlZQQ3Jzyw_LA9R6owj)')
    st.markdown('- [LSTM](https://colab.research.google.com/drive/18NErfQjjzMetzeHoyOnLGGJk0kOEQArx)')
    st.markdown('- [CNN](https://colab.research.google.com/drive/1bmqxciInTWcVNu4Ba92W5L-BLX4ouVaN)')
    st.markdown('- [MLP](https://colab.research.google.com/drive/1X_wSkr5WG7P8vI5BBflmSSMRkpGWrnKJ)')
    st.markdown('- [Comparison: LSTM, BiLSTM, GRU](https://colab.research.google.com/drive/1ydvZQkakrurWONe5naPzf9o4-Mt2yb1J)')
    st.markdown('- [SARIMAX](https://colab.research.google.com/drive/1ze1rbBBhCQNIgyn3svOxja0BAL2WoApv)')
    st.markdown('- [Prophet](https://colab.research.google.com/drive/12YFm4EPOo_kqdwByQ0IKgUOV6pqCiZFD)')
    st.markdown('- [PyCaret RandomForest CV = 3](https://colab.research.google.com/drive/1twzC0-yT-qULYzKjE--La9m9ljuESrYl)')
    st.markdown('- [AutoKeras](https://colab.research.google.com/drive/1i7mF_rJRig2RK01F2lNDuHcgpc5NWB2-)')
    st.markdown('- [Comparison: SARIMAX, Prophet, Bagging-Boosting](https://colab.research.google.com/drive/1oj_8tCobH7yPMMQNXDt6f6q2yB0JfkG7)')

    st.write('Time-Series Classification')
    st.markdown('- [Sequential](https://colab.research.google.com/drive/13jTU4gxV4pAyzLvPZoTqPyUoZRw23lGV)')

    st.write('Time-Series Anomaly Detection')
    st.markdown('- [LSTM](https://colab.research.google.com/drive/1jTvHf5XtAb7a5Qy0DhmWVLDEgcTWbXjw)')

    st.write('More')
    st.markdown('- [Feature Engineering](https://colab.research.google.com/drive/1B1MKdGYdiZA_N7SNlIMP80fa8DCnnyvn)')
    st.markdown('- [Feature Analysis](https://colab.research.google.com/drive/1zmX-D15GbO3fFzJqkyQB59k2sQ2rBrPf)')

    st.write('Clustering & Forecast')
    st.markdown('- [k-NN, LSTM](https://colab.research.google.com/drive/12QQ9P68gNk1RNtmCPPs7INPZSCW7M1TT)')


def geo_loc_cluster():

    st.subheader('Clustering of Geographic Locations')
    d1 = st.date_input("", [datetime.date(2020, 11, 26), datetime.date(2020, 12, 14)])

    st.write(" ")

    st.write("What's done in this project?")
    st.markdown("- Scraped data using BeautifulSoup from multiple data sources.")
    st.markdown("- Used GeoPy library to convert the addresses into coordinates")
    st.markdown("- Utilized Foursquare API to download the names and coordinates of venues like restaurants, parks, rivers, etc. in several parts of the Dhaka division. Combined all the data points and made them usable in Pandas.")
    st.markdown("- One-Hot encoded the venues in Dhaka Division and employed k-means clustering to segment Dhaka division into different zones.")
    st.markdown("- Selected a city and then a neighborhood in that city that has plenty of restaurants, a dense population, and a high literacy rate.")
    st.write(" ")
    st.write("**Libraries and frameworks**: Pandas, Numpy, Missingno, Matplotlib, Sklearn, Folium, Geopy, Urllib, Requests, BeautifulSoup, Foursquare")

    st.write('[See in detail on GitHub](https://github.com/SumaiaParveen/Clustering-Bangladesh-Capital-City-for-Restaurant-Business)')

def nlp_spacy():

    st.subheader('Natural Language Processing with spaCy')
    d1 = st.date_input("", [datetime.date(2020, 9, 15), datetime.date(2020, 10, 13)])

    st.subheader('[Go to the app](https://nlpwithspacy.herokuapp.com/)')
    st.write(" ")
    st.write("What's done in this project?")

    st.markdown("- Tokenization: spaCy segments the text into sentences, words, and punctuation marks. These are called tokens. Texts are tokenized using spaCy model ‘en_core_web_sm’ and visualized in a table as words and punctuations, their Lemmatization, Parts of Speech, and type of the entity.")
    st.markdown("- Word Count: Preprocessed the text (removal of stop-words, punctuations, accented characters, etc.) Computed the word frequency table by computing the number of times each word is present in the document. Finally, the frequency of each word and their percentage are shown in a DataFrame.")
    st.markdown("- Matched Words from Two Documents: spaCy takes inputs from the user, cleans the texts, uses model ‘en_core_web_sm’ to find out Lemmatized words. If the same lemmatized words exist in the two documents, the app shows the words and their frequencies in each document in Tabular format.")
    st.markdown("- Extractive Text Summarization: Five different text summarizers are made available in this app. Gensim, Sumy Lex Rank, Sumy Luhn, Sumy Latent Semantic Analysis, and Sumy Text Rank provide summarized text, time to read the original and summarized document, and simple word clouds generated from the words of the summarized document.")
    st.write("**Libraries and frameworks**: Pandas, Nltk, Spacy, spacy_streamlit, en_core_web_sm, Genism, Sumy, Sklearn, Wordcloud, Matplotlib, Unicode, Pillow, Streamlit, Heroku")
    st.write('[See in detail on GitHub](https://github.com/SumaiaParveen/NLP-with-spaCy/tree/main)')

def bin_class():

    st.subheader('Binary-Classification of Health Conditions')
    d1 = st.date_input("", [datetime.date(2020, 8, 1), datetime.date(2020, 9, 12)])
    st.write("What's done in this project?")

    st.markdown("- Created Binary Classifiers for five different health conditions using several Machine Learning algorithms. Among these five, datasets for various test results and various age groups are available for Cervical Cancer and Dementia, respectively; thus built separate classifiers for different cases.")
    st.markdown("- Employed several feature-selection methods and selected important features to reduce dimension. Handled imbalances of the dataset and took care of outliers and important observations.")
    st.markdown("- Employed Lazypredict and several other Machine Learning models. Examined evaluation metrics like ROC_AUC, Precision-Recall AUC, Precision, Recall, F1 Score for Train and Test Datasets and Accuracy score, and selected a model that gave high Recall (large True Positive and small False Negative values) and F1 Scores.")
    st.markdown("- Plotted confusion matrices, ROC, Precision-Recall curve, and distribution plots of the train and predicted data for each model so the best model is selected.")
    st.markdown("- Employed YellowBrick to visualize Class Prediction Error.")
    st.write("**Libraries and frameworks**: Pandas, Numpy, Missingno, Matplotlib, Seaborn, Sklearn, Scipy, Lazypreict, Xgboost, Lightgbm, Imblearn, Catboost, Yellowbrick, Mlxtend")
    st.write('[See in detail on GitHub](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition)')

def la_aqi():

    st.subheader('Prediction of Air Quality Index of Los Angeles')
    d1 = st.date_input("", [datetime.date(2020, 5, 7), datetime.date(2020, 6, 15)])

    st.subheader('[Go to the app](https://laaqipm25.herokuapp.com/)')
    st.write(" ")
    st.write("What's done in this project?")

    st.markdown("- Created a web app based on a Machine Learning Algorithm that estimates the concentration of Air Quality Index PM 2.5 of Los Angeles, California. This could be useful for someone who is moving to Los Angeles and wants to find out about its air quality.")
    st.markdown("- Scraped climate data from 2010 to 2020 from three climate data websites using Python and Beautiful Soup.")
    st.markdown("- EOptimized Random Forest Regressor using RandomizedsearchCV to reach the best model.")
    
    st.write("**Libraries and frameworks**: Os, Time, Requests, Sys, Csv, Pandas, Numpy, BeautifulSoup, Datetime, Missingno, Matplotlib, Seaborn, Sklearn, Scipy, Lazypreict, Lightgbm, Yellowbrick, Streamlit, Heroku")
    st.write('[See in detail on GitHub](https://github.com/SumaiaParveen/Regression-LA-AQI-Prediction)')

def life_exp():

    st.subheader('Prediction of Life Expectancy from Socio-Economic Factors')
    d1 = st.date_input("", [datetime.date(2020, 3, 8), datetime.date(2020, 4, 28)])

    st.subheader('[Go to the app](https://predictlifespan.herokuapp.com/)')
    st.write(" ")
    st.write("What's done in this project?")

    st.markdown("- Employed LazyPredict to compare the R-Squared and Errors of different models. Optimized ExtraTrees Regressor using RandomizedsearchCV to reach the best model; the base model gave the minimal error.")
    
    st.write("**Libraries and frameworks**: Pandas, Numpy, Missingno, Matplotlib, Seaborn, Sklearn, Scipy, Lazypreict, Yellowbrick, Pillow, Streamlit, Heroku")
    st.write('[See in detail on GitHub](https://github.com/SumaiaParveen/Lifespan-Prediction-using-ExtraTrees)')

#st.sidebar.image("wc (2).png", use_column_width=True)

option = st.sidebar.selectbox('Select from below', ( "Machine Learning Projects", "Passion for Data Analysis", 'Tableau Dashboards', "Flourish Viz"))


###################
# Set up main app #
###################

st.markdown("<h1 style='text-align: center; color: white;'>Project Overview</h1>", unsafe_allow_html=True)

if option == "Machine Learning Projects":

    menu = ["NLP as a Service", "AI for Improving Road Safety in Bangladesh", "Cloud Instance Price Prediction", "Clustering of Geographic Locations", "Natural Language Processing with spaCy",\
    "Binary-Classification of Health Conditions", "Prediction of Air Quality Index of Los Angeles", "Prediction of Life Expectancy from Socio-Economic Factors"]
    choice = st.selectbox("", menu)

    #if choice == 'NLP as a Service':
        #tfg_aws_pred()

    if choice == 'AI for Improving Road Safety in Bangladesh':
        road_safety()

    if choice == 'Cloud Instance Price Prediction':
        tfg_aws_pred()

    if choice == 'Clustering of Geographic Locations':
        geo_loc_cluster()

    if choice == 'Natural Language Processing with spaCy':
        nlp_spacy()

    if choice ==  "Binary-Classification of Health Conditions":
        bin_class()

    if choice ==  "Prediction of Air Quality Index of Los Angeles":
        la_aqi()

    if choice ==  "Prediction of Life Expectancy from Socio-Economic Factors":
        life_exp()



        



if option == "Passion for Data Analysis":

    st.subheader('Data Analysis in Python & R')

    st.write('- [Four Cool Libraries to Automate Data Analysis in Python](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/4-libraries-to-automate-eda-in-3-lines-of.ipynb)')
    st.write('- [Exploratory Data Analysis using DataExplorer](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/auto-data-profiling-by-dataexplorer-breakdowns.ipynb)')
    st.write('- [Immigration to Canada Dataset: Matplotlib Area, Bubble & Waffle Chart](https://github.com/SumaiaParveen/Python-Data-Analysis/blob/main/Immigration_to_Canada_DataViz.ipynb)')
    st.write('- [Analysis of Violence Against Women and Girls (VAWG)](https://github.com/SumaiaParveen/Python-Data-Analysis/blob/main/VAWG%20DataViz.ipynb)')
    st.write('- [Analysis of Vancouver Climate Data](https://github.com/SumaiaParveen/Python-Data-Analysis/blob/main/Van_climate_Viz.ipynb)')
    st.write('- [Heart Failure Data Analysis: Plotly](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/eda-heart-attack-dataset.ipynb)')
    st.write('- [Telecommunication Churn Data Analysis](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/eda-of-telecom-customer-churn-data.ipynb)')
    st.write('- [US Juvenile Crime Data Analysis](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/eda-of-us-juvenile-crime-data.ipynb)')
    st.write('- [US Analyst Job Data Analysis](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/insights-of-job-opportunities-of-analysts-roles.ipynb)')
    st.write('- [Myanmar Coup 2021 Analysis](https://github.com/SumaiaParveen/Python-Data-Analysis/blob/main/myanmarcoup-may-2021.ipynb)')
    st.write('- [Vancouver Crime Data Analysis](https://nbviewer.jupyter.org/github/SumaiaParveen/Python-Data-Analysis/blob/main/vancouver-bc-crime-data-eda.ipynb)')

    

if option == "Tableau Dashboards":

    st.subheader('#1 Sales Analytics Dashboards')

    html_temp = """<div class='tableauPlaceholder' id='viz1627712361794' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sa&#47;SalesAnalytics_16087724524520&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SalesAnalytics_16087724524520&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sa&#47;SalesAnalytics_16087724524520&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627712361794');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='2200px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627714469476' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Su&#47;SuperstoreSalesAnalytics_16087991946110_16277139405220&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SuperstoreSalesAnalytics_16087991946110_16277139405220&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Su&#47;SuperstoreSalesAnalytics_16087991946110_16277139405220&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627714469476');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    st.subheader('#1 HR Analytics Dashboards')

    html_temp = """<div class='tableauPlaceholder' id='viz1627706012707' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USHRAnalytics--AbsencePerformanceScoreSpecialProjectsAccomplished&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='USHRAnalytics--AbsencePerformanceScoreSpecialProjectsAccomplished&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USHRAnalytics--AbsencePerformanceScoreSpecialProjectsAccomplished&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627706012707');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1000px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627706592963' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USHRAnalytics--AverageSalaryResidencyStatusStaffCount&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='USHRAnalytics--AverageSalaryResidencyStatusStaffCount&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USHRAnalytics--AverageSalaryResidencyStatusStaffCount&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627706592963');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1000px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627708117657' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalyticsExperienceAttrition_16277071865030&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HRAnalyticsExperienceAttrition_16277071865030&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalyticsExperienceAttrition_16277071865030&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627708117657');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='750px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627711618390' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalyticsWorkLifeBalanceAttrition&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HRAnalyticsWorkLifeBalanceAttrition&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalyticsWorkLifeBalanceAttrition&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627711618390');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1200px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627710999577' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalytics--TerminationReasons_16277099960210&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HRAnalytics--TerminationReasons_16277099960210&#47;Dashboard' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HRAnalytics--TerminationReasons_16277099960210&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627710999577');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1350px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    html_temp = """<div class='tableauPlaceholder' id='viz1627712131462' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StaffAttrition_16277121242700&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='StaffAttrition_16277121242700&#47;Dashboard2' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StaffAttrition_16277121242700&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1627712131462');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1250px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600)

    st.write("[See more on Tableau Public](https://public.tableau.com/app/profile/sumaia)")


if option == "Flourish Viz":

    st.subheader('#1 Bar chart race showing death count on particular dates over 2016-2021 caused by road crashes.')

    components.html(
        """
        <div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/6687631"><script src="https://public.flourish.studio/resources/embed.js"></script></div>
        """,
        height=700,
    )
    st.subheader('#2 Visualization of statistical snalysis of the road accident datasets.')
    components.html(
        """
        <div class="flourish-embed" data-src="story/941600"><script src="https://public.flourish.studio/resources/embed.js"></script></div>"></script></div>
        """, height=850,
    )



# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------ Summary of Statistics ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------


