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

    st.set_page_config(page_title="PG-LCA-TEA Dashboard", page_icon="🔬", layout="wide")

    st.title("🔬 Phosphogypsum LCA-TEA Framework")
    st.markdown("### Life Cycle Assessment & Techno-Economic Analysis")

    # Sidebar
    st.sidebar.header("Configuration")

    country = st.sidebar.selectbox(
        "Select Country", ["China", "USA", "Morocco", "EU", "Brazil", "India", "Global"]
    )

    pathways = st.sidebar.multiselect(
        "Select Pathways",
        ["PG-SD", "PG-CM", "PG-CB", "PG-SA", "PG-CR", "PG-RE"],
        default=["PG-SD", "PG-CM"],
    )

    functional_unit = st.sidebar.number_input(
        "Functional Unit (tonnes PG)", min_value=1.0, value=1.0, step=1.0
    )

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "LCA Results", "TEA Results", "Comparison", "Live Monitoring"])

    with tab1:
        st.header("Framework Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pathways Selected", len(pathways))
        with col2:
            st.metric("Country", country)
        with col3:
            st.metric("Functional Unit", f"{functional_unit} t PG")

        st.markdown(
            """
        ### Treatment Pathways

        | Code | Pathway | Description | TRL |
        |------|---------|-------------|-----|
        | PG-SD | Stack Disposal | Baseline: engineered stacking | 9 |
        | PG-CM | Cement Production | Cement additive/retarder | 9 |
        | PG-CB | Construction Materials | Bricks, plasterboard | 8 |
        | PG-SA | Soil Amendment | Agricultural application | 8 |
        | PG-CR | Chemical Recovery | (NH₄)₂SO₄ + CaCO₃ | 7 |
        | PG-RE | REE Extraction | Rare earth recovery | 5 |
        """
        )

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

    with tab5:
        st.header("📡 Industrial Live Monitoring")
        st.markdown("Real-time sensor-to-dashboard pipeline streaming from Edge OPC UA to MQTT.")

        # Handle Streamlit fragment decorator for backwards compatibility
        fragment_decorator = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

        def live_monitor_logic():
            import sqlite3
            import pandas as pd
            import os
            
            db_path = "sensors_live.db"
            if not os.path.exists(db_path):
                st.info("Waiting for live telemetry data. Please start edge bridge and stream processor.")
                return

            try:
                # Open with URI so we can enforce read-only
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                df = pd.read_sql(
                    "SELECT timestamp, node_id, value, status, lca_co2_rate, tea_cost_rate FROM telemetry ORDER BY id DESC LIMIT 20",
                    conn
                )
                conn.close()
            except Exception as e:
                st.warning(f"Waiting for telemetry data... ({e})")
                return

            if df.empty:
                st.info("No live telemetry data received yet.")
                return

            # Check for alarms
            alarms = df[df["status"].str.contains("ALARM")]
            if not alarms.empty:
                st.error(f"⚠️ {len(alarms)} constraint violations detected in the recent stream window!")
                st.dataframe(alarms.head())
            else:
                st.success("All systems operating within physical boundaries.")

            # Show metrics for latest entry
            latest = df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Sensor Value", f"{latest['value']:.2f}")
            c2.metric("Instant CO2 Emission Rate", f"{latest['lca_co2_rate']:.2f} kg/s")
            c3.metric("Instant OPEX Rate", f"${latest['tea_cost_rate']:.2f}/s")

            st.subheader("Live Telemetry Stream")
            st.dataframe(df)

        if fragment_decorator:
            # Wrap with run_every 1s
            render_live_monitoring = fragment_decorator(run_every="1s")(live_monitor_logic)
            render_live_monitoring()
        else:
            # Fallback for old streamlit versions
            st.warning("Your Streamlit version does not support st.fragment. Live auto-refresh disabled.")
            if st.button("Refresh Manually"):
                pass
            live_monitor_logic()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PG-LCA-TEA v0.6.5**")
    st.sidebar.markdown("[GitHub](https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum)")


def main():
    run_dashboard()


if __name__ == "__main__":
    main()
