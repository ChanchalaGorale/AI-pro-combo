import streamlit as st
from notebooks.spam_email_detector.utils import predict_email
import spacy
from utils import chat_bot
from notebooks.height_weight_predict.utils import predict_height
from notebooks.car_price_prediction.utils import predict_car_price
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
import numpy as np



nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

# Sidebar contents
st.sidebar.image("static/profile.png", use_column_width=True,)  
st.sidebar.title("Chanchala Gorale")
st.sidebar.markdown("""
                   <h3> <span style="color:#b95d59">AI Engineer</span></h3>
                    """,unsafe_allow_html=True)

st.sidebar.markdown("""Follow Me: [Github](https://github.com/hypothesistribetechnology)   |   [LinkedIn](https://www.linkedin.com/in/chanchala-g-2b040411a/)   |   [Medium](https://medium.com/@cgorale111)""")

# Sidebar buttons
if st.sidebar.button('About Me'):
    st.session_state.page = 'about'
if st.sidebar.button('My Work History'):
    st.session_state.page = 'work'
if st.sidebar.button('My Projects'):
    st.session_state.page = 'projects'
if st.sidebar.button('Connect with Me'):
    st.session_state.page = 'connect'

#st.sidebar.markdown('### [View My Gallery](https://photos.app.goo.gl/42DqydSPLx9R6uid7")')

# manage add button click 
if 'page' not in st.session_state:
    st.session_state.page = 'about'


def set_page(page):
    st.session_state.page = page

# Main content
if st.session_state.page == 'about':
    st.title("About Me")
    st.markdown(
        """
        ### Hello!

        AI Developer with over 7 years of experience in full-stack development and technical leadership. Proven track record in building and managing high-quality web and mobile applications. As the Founder & CTO of Nektor, led product strategy and growth initiatives, resulting in significant market traction and user engagement. 
        
        Currently transitioning into AI engineering, leveraging comprehensive technical skills and extensive project experience to create innovative AI-driven solutions. 

        ### Key Skills
        - **AI Engineer:** Gen AI, Machine Learning, Deep Learning, TensorFlow, Scikit-Learn, Transformer, Flask, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook, LLM, OpenAI, Scapy
        - **Data Handling:** EDA, Feature Engineering, Creating, Validating, Fine-tuning, Deploying Models
        - **Full-Stack Development:** JavaScript, React (Redux), React-Native, Node.js, Express, RESTful APIs, MongoDB/SQL, Testing
        - **Design and Development:** GitHub, Responsive Design, Bootstrap/ANTD/Material UI, Figma, UI/UX Design Principles
        - **Project Management:** JIRA, Slack, Notion, Confluence, Agile Development, Canva, Google Analytics, Meta

        ### Experience

        #### AI Engineer
        - Defined project objectives and deliverables for AI initiatives.
        - Conducted data analysis and preprocessing to create reusable datasets.
        - Developed and fine-tuned machine learning models using TensorFlow, Scikit-Learn, and Transformer architectures.
        - Integrated AI models into applications, ensuring seamless frontend-backend communication.
        - Deployed models using Docker and Kubernetes, with continuous monitoring for reliability.

        #### Product Manager
        - Articulated product vision, strategy, and roadmap.
        - Conducted market research and competitive analysis.
        - Led end-to-end product development lifecycle.
        - Collaborated with cross-functional teams for product delivery.
        - Gathered and analyzed user feedback to drive product improvements.
        - Managed project timelines, resources, and budgets.

        #### Full Stack Developer
        - Developed and maintained full-stack applications using Python (Flask) and JavaScript (React.js and Node.js).
        - Implemented RESTful APIs and optimized application performance.

        #### UI/UX Designer
        - Led design processes using Figma for wireframes, mockups, and prototypes.
        - Conducted user research and usability testing.
        - Developed design systems to ensure consistency.

        #### Technical Writing/Documentation
        - Created comprehensive user guides, API documentation, and technical manuals.
        - Standardized documentation templates and introduced automation tools.
        - Worked closely with developers and QA teams for accurate documentation.

        ### Achievements
        - Launched Nektor.App, achieving 30% repeat user rate by onboarding reputed hospitals across India.
        - Enhanced documentation standards and team performance at Alten, increasing project delivery rate by 20%.
        - Decreased documentation production time by 40% with a single-source publishing system.
        - Successfully built and deployed multiple full-stack applications, improving user experience and meeting client requirements.

        ### Certifications
        - **IBM AI Developer Specialization** (2024, Coursera)
        - **AI Product Management Specialization** (2023, Coursera)
        - **International Certificate in Product Management** (2022, The Institute of Product and Leadership)
        - **Google Agile Project Management** (2022, Coursera)
        - **Full Stack Web Development** (2021, Crampete Institute)
        - **Agile Scrum** (2019, LinkedIn Learning)

        ### Education
        - **Aeronautical Engineering** (2016, Academy of Aviation & Engineering Bangalore)

        Follow Me: [Github](https://github.com/hypothesistribetechnology)   |   [LinkedIn](https://www.linkedin.com/in/chanchala-g-2b040411a/)   |   [Medium](https://medium.com/@cgorale111)
        """
    )

elif st.session_state.page == 'work':
    st.title("My Work History")
    st.markdown(
            f"""
            ### [Nektor](https://nektor.app/)
            Founder & CTO (Sep 2022 - Till Today)

            ### [DarDoc](https://www.dardoc.com/)
            Full-Stack Developer (Aug 2021 - Sep 2022)


            ### [Juvoxa](https://www.linkedin.com/company/juvoxa/people/)
            Front-End Developer (Jan 2021 - Aug 2021)


            ### [Ellucian](https://www.ellucian.com/)
            Sr Technical Writer (June 2019 - Jan 2021)


            ### [Intel](https://www.intel.com/content/www/us/en/homepage.html)
            Product Technical Writer/UI/UX Designer (Nov 2018 - Jun 2019)

            ### [Alten](https://www.alten.com/)
            Technical Writer (Nov 2016 - Nov 2018)
            """
        )
    
elif st.session_state.page == 'projects':
    st.title("My Projects")

    # project 1
    st.markdown(
        """
        ### Chat Bot
        A chatbot created with the Gemini Pro API leverages advanced natural language processing to deliver sophisticated and accurate conversational experiences. It is designed to understand context, provide relevant responses, and handle complex queries efficiently. This AI-powered solution enhances customer interactions, automates support tasks, and improves overall user engagement.
        """
        
    )
    st.button("Chat with Bot", on_click=set_page, args=['chat_bot'])

    # project 1
    st.markdown(
        """
        ### Car Price Prediction
        The Car Price Prediction model uses a RandomForestRegressor to estimate vehicle prices based on various features such as make, model, year, mileage, and more. By leveraging the ensemble learning approach of random forests, the model achieves high accuracy and reliability, making it useful for both buyers and sellers in the automotive market.
        """
        
    )
    st.button("Predict Car Price", on_click=set_page, args=['car_price_predict'])



    # project 1
    st.markdown(
        """
        ### Weight to Height Prediction
        Height prediction based on weight using a Linear Regression model involves finding a linear relationship between weight (independent variable) and height (dependent variable). The model uses this relationship to predict height from given weight data, providing a simple yet effective way to estimate height based on weight measurements.
        """
        
    )
    st.button("Try Height Predictor", on_click=set_page, args=['height_predict'])


    # project 1
    st.markdown(
        """
        ### Spam Email Detector
        The spam email detection app employs a TfidfVectorizer to convert email text into numerical feature vectors and an SVC (Support Vector Classifier) model for classification. This combination effectively distinguishes between spam and legitimate emails, enhancing email security and reducing unwanted messages in users' inboxes.
        """
        
    )
    st.button("Detect Email Spam", on_click=set_page, args=['spam_email_detector'])
    
    # project 2
    # st.markdown(
    #     """
    #     ### Document GPT
    #     Upload your pdf file to get tokens, lemmatize tokens, document summary, and ask question to document

    #     """
        
    # )

    # st.button("Try Doc GPT", on_click=set_page, args=['document_gpt'])


    # project 2
    st.markdown(
        """
        ###  Text analyzer
        Text Analyzer evaluates the sentiment of your text, determining whether it is positive, negative, or neutral. Simply enter your text, and our tool provides an immediate result, helping you understand the emotional tone and potential impact of your message. Perfect for writers, marketers, and anyone needing sentiment analysis.

        """
        
    )
    st.button("Try Text Analyzer", on_click=set_page, args=['sentiment_analyzer'])

    

    # project 2
    st.markdown(
        """
        ###  Image analyzer
        An Image Analyzer is a tool that examines and extracts various features from images, such as objects, colors, shapes, textures, and patterns. It uses advanced algorithms and machine learning techniques to identify and classify elements within the image, enabling detailed analysis and insights for various applications.

        """
        
    )

    st.button("Try Image Analyzer", on_click=set_page, args=['image_analyzer'])
    

    st.markdown(
            f"""
            ### Maleria Detection
            The malaria detection model leverages a Convolutional Neural Network (Conv2D) to analyze microscopic blood smear images. This AI-driven approach enhances the accuracy and speed of diagnosing malaria by identifying parasitic infections within red blood cells. The model automates detection, reducing the need for manual microscopic analysis and supporting timely medical intervention.
            
            [Git Repo](https://github.com/hypothesistribetechnology/malaria-detection/blob/main/prediction%20model.ipynb)

            
            """
        )
    
    st.markdown(
        """
        ### Load Approval Risk Prediction
        Load Approval Risk Prediction involves using LabelEncoder for categorical data encoding, PolynomialFeatures for feature interaction and non-linear transformation, and RandomForestClassifier for robust classification. This combination enhances the model's ability to predict the risk associated with loan approvals by capturing complex relationships within the data.

        [Git Repo](https://github.com/hypothesistribetechnology/load-approval-risk-prediction/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Parkinson's disease detection
        Parkinson's disease detection can be enhanced using machine learning techniques like StandardScaler for feature normalization and Support Vector Classifier (SVC) for classification. StandardScaler standardizes data, improving SVC performance. This method efficiently distinguishes between healthy individuals and those with Parkinson's, aiding early diagnosis and treatment planning.

        [Git Repo](https://github.com/hypothesistribetechnology/parkinson-s-disease-detection/blob/main/model.ipynb)
        """
        
    )

    st.markdown(
        """
        ### Customer segmentation
        Customer segmentation using KMeans clusters clients based on similarities in behavior, demographics, or preferences. By analyzing data, KMeans identifies distinct groups, allowing businesses to tailor marketing strategies and services effectively. This method enhances customer satisfaction and boosts profitability by addressing diverse needs and interests within target segments.

        [Git Repo](https://github.com/hypothesistribetechnology/customers-segmentation/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Mart Sales Prediction
        The "mart-sales-prediction" project employs data preprocessing techniques like StandardScaler for feature scaling and LabelEncoder for categorical variable transformation. It utilizes XGBRegressor for building a predictive model, aiming to forecast sales accurately. This approach ensures robust performance by optimizing data preprocessing and leveraging advanced regression techniques.

        [Git Repo](https://github.com/hypothesistribetechnology/mart-sales-prediction/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Credit Card Fraud 
        Linear regression is inadequate for detecting credit card fraud due to its inability to capture nonlinear patterns in transaction data. Fraud detection demands more sophisticated methods like anomaly detection or machine learning classifiers, which can identify unusual patterns and deviations from normal behavior, crucial for effective fraud prevention.

        [Git Repo](https://github.com/hypothesistribetechnology/creadit-card-fraud/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Gold Price Prediction 
        Using RandomForestRegressor for gold price prediction involves training a model on historical data, incorporating factors like economic indicators and market trends. By leveraging ensemble learning, it enhances prediction accuracy by averaging multiple decision trees. This approach enables forecasting future gold prices based on past patterns and current market conditions.

        [Git Repo](https://github.com/hypothesistribetechnology/gold-price-prediction/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Heart Disease Prediction
        Using Logistic Regression and StandardScaler for heart disease prediction involves preprocessing data with StandardScaler to normalize features, enhancing model performance. Logistic Regression then predicts heart disease based on input data, utilizing a probabilistic approach to classify patients' risk. This method optimizes accuracy by scaling and fitting data appropriately for predictive analysis.

        [Git Repo](https://github.com/hypothesistribetechnology/heart-disease-prediction/blob/main/model.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Load Status Prediction
        The project involves predicting loan status and amounts using Logistic Regression and StandardScaler. Logistic Regression models classify loan approval/rejection, while StandardScaler normalizes numerical features. The approach aims for accurate predictions by fitting data to a logistic curve and scaling inputs for optimal model performance in financial decision-making.

        [Git Repo](https://github.com/hypothesistribetechnology/loan-status-and-load-amount-prediction/blob/main/load%20amount%20prediction.ipynb)
        """
        
    )
    st.markdown(
        """
        ### Forest fire prediction
        Forest fire prediction using LogisticRegression and StandardScaler involves preprocessing data with StandardScaler to normalize features. LogisticRegression then models the likelihood of a fire occurrence based on historical data. This approach aims to classify future fire events, providing early warnings for effective forest management and prevention strategies.

        [Git Repo](https://github.com/hypothesistribetechnology/forest_fire_prediction/blob/main/model.ipynb)
        """
        
    )

    
        
elif st.session_state.page == 'connect':
    st.title("Connect with Me")

    st.markdown(
        """
        I am always eager to connect with like-minded professionals and explore new opportunities in the dynamic world of AI and software development. Let's connect and grow together!


        Follow Me:  [LinkedIn](https://www.linkedin.com/in/chanchala-g-2b040411a/)  |   [GitHub](https://github.com/hypothesistribetechnology)   |   cgorale111@gmail.com  |   [Medium](https://medium.com/@cgorale111)
        """
    )

elif st.session_state.page == 'resume':
    st.title("Download My Resume")
    with open("resume.pdf", "rb") as file:
        btn = st.download_button(
            label="Download Resume",
            data=file,
            file_name="Chanchala Gorale Resume 2024.pdf",
            mime="application/pdf"
        )


# projects
elif st.session_state.page == 'spam_email_detector':
    st.button("← Go Back", on_click=set_page, args=['projects'])

    st.title("Spam Email Detector")
    st.markdown("Enter email to see if the email received is spam or not.")
    email_text = st.text_area("Email Text")

    if st.button("Predict"):
        if email_text:
            prediction = predict_email(email_text)
            st.write(f"The email is: **{prediction}**")
        else:
            st.write("Please enter an email text.")

# elif st.session_state.page == 'document_gpt':
#     docgpt.getgpt(set_page)

elif st.session_state.page == 'chat_bot':
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Chat Bot")
    chat_bot.my_chatbot()

elif st.session_state.page == 'height_predict':
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Weight to Height Prediction")

    text = st.text_area("Enter weight to predict height")

    submit = st.button("Predict Height")

    if text and submit:

        prediction= predict_height(text)

        st.write("Height is about ", prediction)


elif st.session_state.page == 'car_price_predict':
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Car Price Prediction")
    years = st.number_input("Enter total no of yeras since purchase:")
    km = st.number_input("Enter total no km traveled:")
    rating = st.number_input("Enter rating 1-5:")
    condition = st.number_input("Enter rating 1-10:")
    economy = st.number_input("Enter economy:")
    speed = st.number_input("Enter to speed:")
    hp = st.number_input("Enter to HP:")
    torque = st.number_input("Enter to torque:")

    submit = st.button("Predict Price")

    if years and km and rating and condition and economy and speed and hp and torque and submit:

        data=[]

        data.append(years)
        data.append(km)
        data.append(rating)
        data.append(condition)
        data.append(economy)
        data.append(speed)
        data.append(hp)
        data.append(torque)

        print("printing data before",data)

        st.write("Car Data: ", data)

        if len(data)==8:
           
            prediction= predict_car_price(data)

            if prediction:
                st.write("You can sell car for upto: ", prediction)



elif st.session_state.page == 'image_analyzer':
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Image Analyzer")
    image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if image is not None:

        image = Image.open(image)

        resized_image = image.resize((224, 224))

        st.image(resized_image, caption='Resized Image (224x224).', use_column_width=True)

        img_array = np.array(resized_image)

        processed_image= np.expand_dims(img_array, axis=0)
       
        img = tf.keras.applications.mobilenet_v2.preprocess_input(processed_image)

        model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    
        predictions = model.predict(img)

        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        st.write("Image Contains:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}. {label}: {score*100:.2f}%")
            
elif st.session_state.page == 'sentiment_analyzer':
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Text Analyzer")

    text = st.text_area("Enter text to anlyse")

    submit = st.button("Anlyse Sentiment")

    if text and submit:
        sia= SentimentIntensityAnalyzer()

        result =sia.polarity_scores(text)

        sentiments = {
            "neg":"Negative",
            "neu":"Neutral",
            "pos":"Position",
            "compound":"Mixed Emotions"
        }

        for key, value in result.items():
            if value == 1:
                st.write("Sentiment: ", sentiments[key])
                break


    

    