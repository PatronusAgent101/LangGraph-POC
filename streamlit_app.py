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
    
    if st.session_state.selected_tool:
        display_tool_page(st.session_state.selected_tool)
    else:
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
        
        # # Check if a tool has been selected
        # if 'selected_tool' in st.session_state:
        #     display_tool_page(st.session_state.selected_tool)

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
    st.title(f"{tool_name}")
    
    # Add details specific to Tool 1
    if tool_name == "Tool 1":
        st.write("""
        This tool performs quadratic equation calculations based on values from your uploaded file.
        Upload a text file containing values a, b, and c (one per line) to solve the equation: ax² + bx + c = 0
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a text file with a, b, c values (one per line)", type=["txt"])
        
        # Process the file if uploaded
        if uploaded_file is not None:
            if st.button("Submit"):
                # Show a progress bar
                progress = st.progress(0)
                status_text = st.empty()
                
                # Process the file with progress indicators
                try:
                    # Update progress
                    for i in range(101):
                        # Update progress bar
                        progress.progress(i)
                        if i < 30:
                            status_text.text("Reading file...")
                        elif i < 60:
                            status_text.text("Processing values...")
                        elif i < 90:
                            status_text.text("Calculating results...")
                        else:
                            status_text.text("Finalizing...")
                        time.sleep(0.02)
                    
                    # Read and process the file
                    content = uploaded_file.getvalue().decode("utf-8")
                    a, b, c = parse_file_content(content)
                    
                    # Display the results
                    st.subheader("Results")
                    st.write(f"Parsed values: a = {a}, b = {b}, c = {c}")
                    
                    # Calculate using quadratic formula
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant < 0:
                        st.write("The equation has no real solutions.")
                    elif discriminant == 0:
                        x = -b / (2*a)
                        st.write(f"The equation has one solution: x = {x}")
                    else:
                        x1 = (-b + (discriminant)**0.5) / (2*a)
                        x2 = (-b - (discriminant)**0.5) / (2*a)
                        st.write(f"The equation has two solutions:")
                        st.write(f"x₁ = {x1}")
                        st.write(f"x₂ = {x2}")
                    
                    # Visualize the quadratic function
                    st.subheader("Visualization")
                    st.write("Equation: {}x² + {}x + {} = 0".format(a, b, c))
                    
                    # Show equation in a more readable format
                    equation_str = f"{a}x² "
                    if b >= 0:
                        equation_str += f"+ {b}x "
                    else:
                        equation_str += f"- {abs(b)}x "
                    
                    if c >= 0:
                        equation_str += f"+ {c} = 0"
                    else:
                        equation_str += f"- {abs(c)} = 0"
                    
                    st.latex(equation_str.replace("x²", "x^2"))
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    
                # Clear progress indicators
                progress.empty()
                status_text.empty()
    
    else:
        # Generic content for other tools
        st.write(f"This is the detailed view for {tool_name}.")
    
    # Add a back button
    if st.button("Back to Agent Catalogue"):
        st.session_state.selected_tool = None
        st.rerun()

def parse_file_content(content):
    """Parse the uploaded file content to extract a, b, c values."""
    lines = content.strip().split('\n')
    
    # Try to find values in the first 3 lines
    values = []
    for line in lines[:3]:
        # Look for numbers (integers or floats)
        matches = re.findall(r'-?\d+\.?\d*', line)
        if matches:
            values.append(float(matches[0]))
    
    # Check if we got all three values
    if len(values) != 3:
        raise ValueError("File must contain 3 numeric values (a, b, c)")
    
    return values[0], values[1], values[2]
    
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
