import streamlit as st
from supabase import create_client, Client


#this is an instance of database which helps us querying the database
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_PUBLISHABLE_KEY"]
)