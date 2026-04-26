st.markdown(f"""
<style>
.reportview-container .main .block-container{{
    max-width: 1200px;
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {bucknell_navy};
}}

/* 🔥 REMOVE ALL LINK-LIKE BEHAVIOR */
a {{
    text-decoration: none !important;
    color: inherit !important;
}}

a:hover {{
    text-decoration: none !important;
    color: inherit !important;
}}

/* 🔥 FORCE TEXT INSIDE MARKDOWN BLOCKS TO BE NON-INTERACTIVE */
div, p, span {{
    text-decoration: none !important;
}}

h5 {{
    text-decoration: none !important;
    cursor: default !important;
}}

.stButton>button {{
    background-color: {bucknell_orange};
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: 1px solid {bucknell_orange};
}}

.stButton>button:hover {{
    background-color: {bucknell_navy};
    color: white;
    border: 1px solid {bucknell_navy};
}}

.stAlert {{
    border-radius: 5px;
}}

.stAlert.info {{
    background-color: {bucknell_navy};
    color: white;
}}

.stAlert.success {{
    background-color: {bucknell_orange};
    color: white;
}}

.stAlert.error {{
    background-color: #A30000;
    color: white;
}}
</style>
""", unsafe_allow_html=True)
