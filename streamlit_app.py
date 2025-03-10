import streamlit as st

def main():
    # Set page configuration
    st.set_page_config(page_title="Agent Catalog", layout="wide")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        selected_tab = st.radio(
            "Select a page:",
            ["Agent Catalogue", "Agent 1", "Agent 2", "About Product"]
        )
    
    # Display content based on selected tab
    if selected_tab == "Agent Catalogue":
        display_agent_catalogue()
    elif selected_tab == "Agent 1":
        display_agent_1()
    elif selected_tab == "Agent 2":
        display_agent_2()
    elif selected_tab == "About Product":
        display_about_product()

def display_agent_catalogue():
    # Header
    st.title("Agent Cataouge")
    st.write("Select from the agents and products below:")
    
    # Display user login info
    st.markdown(
        """
        <div style="background-color: #f8f9fa; padding: 8px 15px; border-radius: 5px; width: fit-content;">
            Sankalp Saklecha (saklecha.02@gmail.com) is signed in
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create two rows with three clickable cards each
    col1, col2, col3 = st.columns(3)
    
    # First row of clickable cards
    with col1:
        if create_clickable_card("CLICKABLE LINK 1"):
            st.session_state.selected_tool = "Tool 1"
            st.experimental_rerun()
            
    with col2:
        if create_clickable_card("CLICKABLE LINK 2"):
            st.session_state.selected_tool = "Tool 2"
            st.experimental_rerun()
            
    with col3:
        if create_clickable_card("CLICKABLE LINK 3"):
            st.session_state.selected_tool = "Tool 3"
            st.experimental_rerun()
    
    # Second row of clickable cards
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if create_clickable_card("CLICKABLE LINK 4"):
            st.session_state.selected_tool = "Tool 4"
            st.experimental_rerun()
            
    with col5:
        if create_clickable_card("CLICKABLE LINK 5"):
            st.session_state.selected_tool = "Tool 5"
            st.experimental_rerun()
            
    with col6:
        if create_clickable_card("CLICKABLE LINK 6"):
            st.session_state.selected_tool = "Tool 6"
            st.experimental_rerun()
    
    # Check if a tool has been selected
    if 'selected_tool' in st.session_state:
        display_tool_page(st.session_state.selected_tool)

def create_clickable_card(title):
    # Create a clickable card with a border
    card_html = f"""
    <div style="
        border: 1px solid black;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        cursor: pointer;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <h3>{title}</h3>
    </div>
    """
    
    # Use a container for the clickable area
    container = st.container()
    container.markdown(card_html, unsafe_allow_html=True)
    
    # Check if the container was clicked
    return container.button("", key=f"btn_{title}", help=f"Click to open {title}")

def display_tool_page(tool_name):
    st.subheader(f"{tool_name} Details")
    st.write(f"This is the detailed view for {tool_name}.")
    
    # Add a back button
    if st.button("Back to Agent Catalogue"):
        st.session_state.pop('selected_tool', None)
        st.experimental_rerun()

def display_agent_1():
    st.title("Agent 1")
    st.write("This is the Agent 1 page content.")

def display_agent_2():
    st.title("Agent 2")
    st.write("This is the Agent 2 page content.")

def display_about_product():
    st.title("About Product")
    st.write("This page contains information about the product.")

if __name__ == "__main__":
    main()
