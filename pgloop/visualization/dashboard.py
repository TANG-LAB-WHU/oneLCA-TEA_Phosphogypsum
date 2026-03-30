"""
Streamlit Dashboard for PG-LCA-TEA

Run with: streamlit run src/visualization/dashboard.py
"""


def run_dashboard():
    """Main entry point for the dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return
    
    st.set_page_config(
        page_title="PG-LCA-TEA Dashboard",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Phosphogypsum LCA-TEA Framework")
    st.markdown("### Life Cycle Assessment & Techno-Economic Analysis")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    country = st.sidebar.selectbox(
        "Select Country",
        ["China", "USA", "Morocco", "EU", "Brazil", "India", "Global"]
    )
    
    pathways = st.sidebar.multiselect(
        "Select Pathways",
        ["PG-SD", "PG-CM", "PG-CB", "PG-SA", "PG-CR", "PG-RE"],
        default=["PG-SD", "PG-CM"]
    )
    
    functional_unit = st.sidebar.number_input(
        "Functional Unit (tonnes PG)",
        min_value=1.0,
        value=1.0,
        step=1.0
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "LCA Results", "TEA Results", "Comparison"])
    
    with tab1:
        st.header("Framework Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pathways Selected", len(pathways))
        with col2:
            st.metric("Country", country)
        with col3:
            st.metric("Functional Unit", f"{functional_unit} t PG")
        
        st.markdown("""
        ### Treatment Pathways
        
        | Code | Pathway | Description | TRL |
        |------|---------|-------------|-----|
        | PG-SD | Stack Disposal | Baseline: engineered stacking | 9 |
        | PG-CM | Cement Production | Cement additive/retarder | 9 |
        | PG-CB | Construction Materials | Bricks, plasterboard | 8 |
        | PG-SA | Soil Amendment | Agricultural application | 8 |
        | PG-CR | Chemical Recovery | (NH₄)₂SO₄ + CaCO₃ | 7 |
        | PG-RE | REE Extraction | Rare earth recovery | 5 |
        """)
    
    with tab2:
        st.header("Life Cycle Assessment")
        st.info("Select pathways and click 'Run Analysis' to see LCA results.")
        
        if st.button("Run LCA Analysis", key="lca"):
            with st.spinner("Calculating..."):
                st.success("LCA calculation complete!")
                # Placeholder for actual results
                st.bar_chart({"Climate Change": [10, 8, 12, 5, 7, 15]})
    
    with tab3:
        st.header("Techno-Economic Analysis")
        st.info("Select pathways and click 'Run Analysis' to see TEA results.")
        
        if st.button("Run TEA Analysis", key="tea"):
            with st.spinner("Calculating..."):
                st.success("TEA calculation complete!")
                # Placeholder for actual results
                st.bar_chart({"CLCC ($/t)": [15, 25, 30, 10, 45, 80]})
    
    with tab4:
        st.header("Pathway Comparison")
        st.markdown("Compare environmental and economic performance across pathways.")
        
        st.pyplot(fig=None)  # Placeholder
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**PG-LCA-TEA v0.2.0**")
    st.sidebar.markdown("[GitHub](https://github.com/yourusername/PG_ucLCA-TEA)")


if __name__ == "__main__":
    run_dashboard()
