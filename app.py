import streamlit as st

# Sidebar contents
st.sidebar.image("static/profile.png", use_column_width=True)  
st.sidebar.title("Chanchala Gorale")
st.sidebar.markdown("""
                   <h3> <span style="color:#b95d59">AI Engineer</span></h3>
                    """,unsafe_allow_html=True)

# Sidebar buttons
if st.sidebar.button('About Me'):
    st.session_state.page = 'about'
if st.sidebar.button('My Work History'):
    st.session_state.page = 'projects'
if st.sidebar.button('My Projects'):
    st.session_state.page = 'projects'
if st.sidebar.button('Connect with Me'):
    st.session_state.page = 'connect'
if st.sidebar.button('Download My Resume'):
    st.session_state.page = 'resume'
if st.sidebar.button('Gallery'):
    st.session_state.page = 'photo'



# manage add button click 
if 'page' not in st.session_state:
    st.session_state.page = 'about'


    # Main content
if st.session_state.page == 'about':
    st.title("About Me")
    st.markdown(
        """
        ### Hello!

        I am a AI Developer with over 7 years of experience in full-stack development and technical leadership. Proven track record in building and managing high-quality web and mobile applications. As the Founder & CTO of Nektor, led product strategy and growth initiatives, resulting in significant market traction and user engagement. 
        
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

elif st.session_state.page == 'projects':
    st.title("My Projects")
    st.markdown(
            f"""
            ### Project 1
            Project description goes here
            
            [Try Now ](https://github.com)

            """
        )
        

elif st.session_state.page == 'connect':
    st.title("Connect with Me")
    st.markdown(
        """
        [LinkedIn](https://linkedin.com/in/username) \\
        [GitHub](https://github.com/username) 
        [Twitter](https://twitter.com/username)
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

elif st.session_state.page == 'photo':
    st.title("My Gallery")
    for i in range(1, 71):
        st.image(f"static/gallery/{i}.jpg", use_column_width=True)  

   