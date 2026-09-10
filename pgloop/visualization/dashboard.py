"""
Streamlit Dashboard for PG-LCA-TEA (v0.7.0)

Interactive, real-time decision dashboard powered by IntegratedAssessmentEngine.
Run with: streamlit run pgloop/visualization/dashboard.py
"""

from typing import Dict, List, Any


def run_dashboard():
    """Main entry point for the interactive Streamlit dashboard."""
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        print("Streamlit or Pandas not installed. Run: pip install streamlit pandas")
        return

    from pgloop.assessment import IntegratedAssessmentEngine
    from pgloop.pathways import PATHWAYS, list_pathways, get_pathway
    from pgloop.iodata import DataHub

    st.set_page_config(
        page_title="PhosphogypsumBot Dashboard v0.7.0",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 PhosphogypsumBot: Industrial LCA-TEA Framework")
    st.markdown(
        "**Physics-Informed Decision Intelligence for Industrial Phosphogypsum Engineering** &nbsp;|&nbsp; "
        "`v0.7.0` &nbsp;|&nbsp; [GitHub](https://github.com/TANG-LAB-WHU/oneLCA-TEA_Phosphogypsum)"
    )

    # -------------------------------------------------------------------------
    # Sidebar Configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Configuration")

    country = st.sidebar.selectbox(
        "Target Country / Jurisdiction",
        ["China", "USA", "Morocco", "EU", "Brazil", "India", "Global"],
        index=0,
    )

    all_codes = list_pathways()
    pathway_label_map = {
        code: f"{code} ({get_pathway(code).name})" for code in all_codes
    }

    selected_codes = st.sidebar.multiselect(
        "Select Valorization Pathways",
        options=all_codes,
        default=["PG-Stack", "PG-CementProd", "PG-ConstructMat", "PG-SulfurAcid"],
        format_func=lambda x: pathway_label_map.get(x, x),
    )

    functional_unit = st.sidebar.number_input(
        "Functional Unit (tonnes PG)",
        min_value=0.1,
        max_value=1000000.0,
        value=1.0,
        step=1.0,
        help="Base functional unit mass of phosphogypsum to be treated.",
    )

    fu_kg = float(functional_unit * 1000.0)

    # Initialize computational engine
    engine = IntegratedAssessmentEngine(country=country)

    # -------------------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "🌱 LCA Footprint",
        "💰 TEA Economics",
        "⚖️ MCDA Ranking",
        "📡 Live Telemetry",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Overview
    # -------------------------------------------------------------------------
    with tab1:
        st.header("System Overview & Registered Pathways")

        c1, c2, c3 = st.columns(3)
        c1.metric("Pathways Selected", f"{len(selected_codes)} / {len(all_codes)}")
        c2.metric("Jurisdiction", country)
        c3.metric("Functional Unit", f"{functional_unit:,.1f} t PG ({fu_kg:,.0f} kg)")

        st.markdown("### Registered Valorization Pathways")
        pathway_rows = []
        for code in all_codes:
            pw = get_pathway(code)
            pathway_rows.append({
                "Code": pw.code,
                "Pathway Name": pw.name,
                "Category": getattr(pw, "category", "Valorization"),
                "TRL": getattr(pw, "trl", 7),
                "Primary Products": ", ".join(getattr(pw, "products", ["By-products"])),
            })
        st.dataframe(pd.DataFrame(pathway_rows), use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 2: Life Cycle Assessment (LCA)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("🌱 Environmental Life Cycle Assessment (ISO 14040)")

        if not selected_codes:
            st.warning("Please select at least one pathway in the sidebar.")
        else:
            if st.button("🚀 Calculate Forward LCA", key="btn_lca", type="primary"):
                with st.spinner("Computing forward LCA footprints across selected pathways..."):
                    lca_records = []
                    for code in selected_codes:
                        res = engine.assess(code, functional_unit_kg=fu_kg)
                        impacts = res["lca"]["impacts"]
                        lca_records.append({
                            "Pathway": res["pathway_name"],
                            "Code": code,
                            "Climate Change (kg CO2-eq)": impacts.get("climate_change", 0.0),
                            "Acidification (kg SO2-eq)": impacts.get("acidification", 0.0),
                            "Eutrophication Fresh (kg P-eq)": impacts.get("eutrophication_fresh", 0.0),
                            "Human Toxicity Cancer (CTUh)": impacts.get("human_toxicity_cancer", 0.0),
                        })

                    df_lca = pd.DataFrame(lca_records)
                    st.session_state["df_lca"] = df_lca

            if "df_lca" in st.session_state:
                df_lca = st.session_state["df_lca"]
                st.subheader("Carbon Footprint Comparison (GWP)")
                chart_data = df_lca.set_index("Pathway")[["Climate Change (kg CO2-eq)"]]
                st.bar_chart(chart_data)

                st.subheader("Detailed Midpoint Environmental Impacts")
                st.dataframe(df_lca, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 3: Techno-Economic Analysis (TEA)
    # -------------------------------------------------------------------------
    with tab3:
        st.header("💰 Techno-Economic Analysis & Life Cycle Costing")

        if not selected_codes:
            st.warning("Please select at least one pathway in the sidebar.")
        else:
            if st.button("🚀 Calculate TEA Metrics", key="btn_tea", type="primary"):
                with st.spinner("Calculating CAPEX, OPEX, CLCC, and External Environmental Costs..."):
                    tea_records = []
                    for code in selected_codes:
                        res = engine.assess(code, functional_unit_kg=fu_kg)
                        tea = res["tea"]
                        tea_records.append({
                            "Pathway": res["pathway_name"],
                            "Code": code,
                            "CLCC ($/t)": tea["clcc"],
                            "SLCC ($/t)": tea["slcc"],
                            "Annualized CAPEX ($)": tea["capex_annualized"],
                            "Annual OPEX ($)": tea["opex_total"],
                            "Gross Revenue ($)": tea["revenue"],
                            "NPV ($)": tea["npv"],
                            "Payback (years)": tea["payback_years"],
                        })

                    df_tea = pd.DataFrame(tea_records)
                    st.session_state["df_tea"] = df_tea

            if "df_tea" in st.session_state:
                df_tea = st.session_state["df_tea"]
                st.subheader("Conventional Life Cycle Cost (CLCC vs SLCC)")
                chart_data = df_tea.set_index("Pathway")[["CLCC ($/t)", "SLCC ($/t)"]]
                st.bar_chart(chart_data)

                st.subheader("Financial Metrics Breakdown")
                st.dataframe(df_tea, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 4: MCDA Ranking & Comparison
    # -------------------------------------------------------------------------
    with tab4:
        st.header("⚖️ Multi-Criteria Decision Analysis (MCDA 5D TEPES)")

        if not selected_codes:
            st.warning("Please select at least one pathway in the sidebar.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            w_env = c1.slider("Environmental Weight", 0.0, 1.0, 0.30, step=0.05)
            w_econ = c2.slider("Economic Weight", 0.0, 1.0, 0.40, step=0.05)
            w_risk = c3.slider("Risk Weight", 0.0, 1.0, 0.15, step=0.05)
            w_soc = c4.slider("Social Weight", 0.0, 1.0, 0.15, step=0.05)

            if st.button("⚖️ Run TOPSIS / VIKOR Ranking", key="btn_rank", type="primary"):
                with st.spinner("Performing multi-criteria ranking and Pareto frontier optimization..."):
                    recommendations = engine.rank_all_pathways(
                        pathway_codes=selected_codes,
                        functional_unit_kg=fu_kg,
                        weights={
                            "environmental": w_env,
                            "economic": w_econ,
                            "risk": w_risk,
                            "social": w_soc,
                        },
                    )

                    rank_records = [
                        {
                            "Rank": r.rank,
                            "Pathway": r.pathway_name,
                            "Composite Score": round(r.score, 4),
                            "Pareto Optimal": "✅ Yes" if r.is_pareto_optimal else "❌ Dominated",
                            "Strengths": ", ".join(r.strengths) if r.strengths else "N/A",
                            "Weaknesses": ", ".join(r.weaknesses) if r.weaknesses else "N/A",
                            "Explanation": r.explanation,
                        }
                        for r in recommendations
                    ]
                    st.session_state["df_rank"] = pd.DataFrame(rank_records)

            if "df_rank" in st.session_state:
                df_rank = st.session_state["df_rank"]
                st.subheader("Optimal Pathway Recommendations")
                st.dataframe(df_rank, use_container_width=True)

                st.subheader("MCDA Composite Scores")
                st.bar_chart(df_rank.set_index("Pathway")[["Composite Score"]])

    # -------------------------------------------------------------------------
    # Tab 5: Industrial Live Monitoring
    # -------------------------------------------------------------------------
    with tab5:
        st.header("📡 Industrial IoT Live Telemetry Stream")
        st.markdown("Real-time edge sensor streaming from Edge OPC UA ➔ MQTT ➔ SQLite WAL database.")

        fragment_decorator = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

        def live_monitor_logic():
            import sqlite3
            import os

            db_paths = [
                "sensors_live.db",
                str(DataHub.TELEMETRY_DB),
                str(DataHub.RAW_TELEMETRY / "sensors_live.db"),
            ]
            active_db = next((p for p in db_paths if os.path.exists(p)), None)

            if not active_db:
                st.info(
                    "Waiting for live telemetry database (`sensors_live.db`). "
                    "Start `EdgeBridge` or `StreamProcessor` to stream live OPC UA sensors."
                )
                return

            try:
                conn = sqlite3.connect(f"file:{active_db}?mode=ro", uri=True)
                df = pd.read_sql(
                    "SELECT timestamp, node_id, value, status, lca_co2_rate, tea_cost_rate "
                    "FROM telemetry ORDER BY id DESC LIMIT 20",
                    conn,
                )
                conn.close()
            except Exception as e:
                st.warning(f"Waiting for telemetry data... ({e})")
                return

            if df.empty:
                st.info("No live telemetry rows received yet.")
                return

            alarms = df[df["status"].str.contains("ALARM", na=False)]
            if not alarms.empty:
                st.error(f"⚠️ {len(alarms)} constraint violations detected in the recent stream window!")
                st.dataframe(alarms.head(), use_container_width=True)
            else:
                st.success("All systems operating within physical boundaries.")

            latest = df.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Sensor Reading", f"{latest['value']:.2f}")
            col2.metric("Instant CO2 Rate", f"{latest['lca_co2_rate']:.2f} kg/s")
            col3.metric("Instant OPEX Rate", f"${latest['tea_cost_rate']:.2f}/s")

            st.subheader("Recent Stream Buffer")
            st.dataframe(df, use_container_width=True)

        if fragment_decorator:
            render_live_monitoring = fragment_decorator(run_every="1s")(live_monitor_logic)
            render_live_monitoring()
        else:
            if st.button("🔄 Refresh Telemetry"):
                pass
            live_monitor_logic()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PhosphogypsumBot v0.7.0**")
    st.sidebar.markdown("TANG Lab at Wuhan University")


def main():
    run_dashboard()


if __name__ == "__main__":
    main()

