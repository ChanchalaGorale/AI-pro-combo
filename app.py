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
import webbrowser



nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

# Sidebar contents
st.sidebar.image("static/profile.png", use_column_width=True,)  
st.sidebar.title("Chanchala Gorale")
st.sidebar.markdown("""
                   <h3> <span style="color:#b95d59">AI Engineer</span></h3>
                    """,unsafe_allow_html=True)

# Sidebar buttons
if st.sidebar.button('About Me'):
    st.session_state.page = 'about'
if st.sidebar.button('My Work History'):
    st.session_state.page = 'work'
if st.sidebar.button('My Projects'):
    st.session_state.page = 'projects'
if st.sidebar.button('Download My Resume'):
    st.session_state.page = 'resume'
if st.sidebar.button('Connect with Me'):
    st.session_state.page = 'connect'
if st.sidebar.button('My Gallery'):
    link="https://photos.app.goo.gl/42DqydSPLx9R6uid7"
    webbrowser.open(link, new=2)
    #st.session_state.page = 'photo'

# manage add button click 
if 'page' not in st.session_state:
    st.session_state.page = 'projects'


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

        [Github](https://github.com/hypothesistribetechnology) [LinkedIn](https://www.linkedin.com/in/chanchala-g-2b040411a/)
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
        Ask question & get answer using google gemini pro chatbot api
        """
        
    )
    st.button("Chat with Bot", on_click=set_page, args=['chat_bot'])

    # project 1
    st.markdown(
        """
        ### Car Price Prediction
        some description
        """
        
    )
    st.button("Predict Car Price", on_click=set_page, args=['car_price_predict'])



    # project 1
    st.markdown(
        """
        ### Weight to Height Prediction
        some content
        """
        
    )
    st.button("Try Height Predictor", on_click=set_page, args=['height_predict'])


    # project 1
    st.markdown(
        """
        ### Spam Email Detector
        Upload your email pdf file to see if the email received in spam or not.
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
        Anlyse sentiment of the twitter text

        """
        
    )
    st.button("Try Text Analyzer", on_click=set_page, args=['sentiment_analyzer'])

    

    # project 2
    st.markdown(
        """
        ###  Image analyzer
        Anlyse sentiment of the twitter text

        """
        
    )

    st.button("Try Image Analyzer", on_click=set_page, args=['image_analyzer'])
    

    st.markdown(
            f"""
            ### Recommendation system
            The Movie Recommendation System project involves designing an AI algorithm that suggests movies to users based on their preferences and viewing history.

            ### Chatbot for Customer Service
            Utilizing natural language processing (NLP) and machine learning algorithms, these chatbots can significantly improve the efficiency and availability of customer service across various industries.
            
            ### Stock price prediction
            Stock Price Prediction projects use machine learning algorithms to forecast stock prices based on historical data.

            ### Autonomous driving system
            An Autonomous Driving System represents a middle-ground AI project, focusing on enabling vehicles to navigate and operate without human intervention. These systems can interpret sensory information by leveraging sensors, cameras, and complex AI algorithms to identify appropriate navigation paths, obstacles, and relevant signage. 

            ### Handwritten digit recognition
            The Handwritten Digit Recognition project is a foundational application of computer vision that involves training a machine learning model to identify and classify handwritten digits from images.

            ### Image manipulation
            Computer Vision combines machine learning with image/vision analysis to enable systems to infer insights from videos and images. For a computer, it becomes quite a challenge to interpret pictures and distinguish the features. 

            ### Instagram spam detection
            A fraud detection system employs machine learning algorithms to identify fraudulent activities in transactions, such as in banking or online retail. This project involves analyzing patterns and anomalies in transaction data to flag potentially fraudulent operations for further investigation.

            ### Mask detection
            Employ computer vision techniques like convolutional neural networks (CNNs) to develop a model capable of distinguishing between masked and unmasked faces. 

            ### Object detection
            Object Detection. Similar to that of face detection, objects can also be analysed using AI to determine the specifics or type of object it is.

            ### Translation
            Translation is the communication of the meaning of a source-language text by means of an equivalent target-language text.

            ### Face recognition
            Face Recognition. To create a face recognition system, one can start by collecting a dataset of images containing faces. 

            ### Fraud detection system
            An Advanced Fraud Detection System uses AI to identify potentially fraudulent transactions in real-time, minimizing financial losses and enhancing security. 

            ### Music recommendation
            Music Recommendation System. Budding AI developers can create music recommendation systems that are built upon music and genre datasets.

            ### Voice assistant
            Bing-GPT Voice Assistant. Build your own AI-powered personal assistant just like JARVIS. 

            ### Email generator
            With LLMs, you can build an email generator that takes a few prompts and magically generates engaging and personalized emails.

            ### Animal species prediction
            Prediction of Bird Species. Utilizing machine learning algorithms, this project predicts bird species based on images or audio recordings.

            ### Heart disease prediction
            Heart Disease Prediction Project. This project is beneficial from the medical perspective since it is designed to provide online medical consultati
            """
        )
        
elif st.session_state.page == 'connect':
    st.title("Connect with Me")

    st.markdown(
        """
        I am always eager to connect with like-minded professionals and explore new opportunities in the dynamic world of AI and software development. Let's connect and grow together!


        [LinkedIn](https://www.linkedin.com/in/chanchala-g-2b040411a/)  |   [GitHub](https://github.com/hypothesistribetechnology)   |   cgorale111@gmail.com  |   [Medium](https://medium.com/@cgorale111)
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

# elif st.session_state.page == 'photo':
#     st.title("My Gallery")
#     for i in range(1, 71):
#         st.image(f"static/gallery/{i}.jpg", use_column_width=True)  

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


    

    