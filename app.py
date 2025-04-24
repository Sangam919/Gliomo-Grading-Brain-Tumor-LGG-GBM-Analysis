import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Glioma Mutations Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek, interactive design
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fb; animation: fadeIn 1.2s ease-in;}
    .sidebar .sidebar-content {background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-family: 'Inter', sans-serif; font-weight: 700;}
    h2 {color: #3b82f6; font-family: 'Inter', sans-serif; font-size: 1.5em;}
    .stButton>button {
        background: linear-gradient(to right, #4f46e5, #3b82f6);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #4338ca, #2563eb);
        transform: translateY(-2px);
    }
    .stSelectbox, .stMultiselect {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 10px;
    }
    .stDataFrame {border: 1px solid #e2e8f0; border-radius: 12px; background-color: #ffffff;}
    .expander {border: 1px solid #e2e8f0; border-radius: 12px; background-color: #ffffff; padding: 20px; margin-bottom: 15px;}
    .metric-card {background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; transition: transform 0.3s;}
    .metric-card:hover {transform: translateY(-5px);}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load data
@st.cache_data
def load_data():
    try:
        file_path = Path("TCGA_GBM_LGG_Mutations_all.csv")
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Function to parse age
def parse_age(age_str):
    try:
        if isinstance(age_str, str) and age_str.strip():
            years = int(age_str.split(" ")[0])
            return years
        return None
    except:
        return None

# Load the dataset
df = load_data()

if df is not None:
    # Title and guided intro
    st.title("🧬 Glioma Mutations Explorer")
    st.markdown(
        """
        Welcome to your journey through Glioblastoma (GBM) and Low-Grade Glioma (LGG) mutations! 
        Use the sidebar to pick a tumor type, then explore charts below. Click expanders to dive in—hover for details!
        """
    )

    # Minimal Sidebar
    st.sidebar.header("🔎 Filter")
    with st.sidebar:
        st.markdown("Choose a tumor type:")
        grade_options = ["All", "LGG", "GBM"]
        selected_grade = st.selectbox(
            "Tumor Type", 
            grade_options, 
            help="Pick GBM, LGG, or All to explore."
        )

    # Apply filter
    filtered_df = df.copy()
    if selected_grade != "All":
        filtered_df = filtered_df[filtered_df["Grade"] == selected_grade]
    filtered_df["Age_years"] = filtered_df["Age_at_diagnosis"].apply(parse_age)

    # Gene Selection (moved to main panel)
    mutation_cols = [col for col in df.columns if col not in [
        "Grade", "Project", "Case_ID", "Gender", "Age_at_diagnosis", 
        "Primary_Diagnosis", "Race"
    ]]
    top_genes = ["IDH1", "TP53", "ATRX", "PTEN", "EGFR"]
    st.subheader("🎯 Pick Genes to Explore")
    col1, col2 = st.columns(2)
    with col1:
        gene_1 = st.selectbox("First Gene", top_genes, index=0, key="gene1")
    with col2:
        gene_2 = st.selectbox("Second Gene (for comparison)", top_genes, index=1, key="gene2")

    # Metrics with animation
    st.subheader("📊 Snapshot")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Cases", len(filtered_df))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("LGG Cases", len(filtered_df[filtered_df["Grade"] == "LGG"]))
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("GBM Cases", len(filtered_df[filtered_df["Grade"] == "GBM"]))
        st.markdown("</div>", unsafe_allow_html=True)

    # Collapsible Sections
    with st.expander("📋 Data Peek", expanded=True):
        st.markdown("**Check out the dataset.**")
        st.dataframe(filtered_df[[gene_1, gene_2, "Grade", "Age_years", "Primary_Diagnosis"]].head(15), height=200)
        st.write(f"**Total Cases**: {len(filtered_df)}")

    with st.expander(f"🧬 Genes: {gene_1} vs {gene_2}"):
        st.markdown(f"**Compare {gene_1} and {gene_2} mutations.**")
        compare_df = pd.DataFrame({
            "Gene": [gene_1, gene_2],
            "Mutation Rate (%)": [
                (filtered_df[gene_1] == "MUTATED").mean() * 100,
                (filtered_df[gene_2] == "MUTATED").mean() * 100
            ]
        })
        fig_compare = px.bar(
            compare_df, x="Gene", y="Mutation Rate (%)", 
            title=f"{gene_1} vs {gene_2} Mutation Rates",
            color="Mutation Rate (%)", color_continuous_scale="Blues",
            text_auto=".1f"
        )
        fig_compare.update_traces(textposition="outside")
        st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("🎂 Ages"):
        st.markdown("**See patient ages.**")
        age_data = filtered_df["Age_years"].dropna()
        if not age_data.empty:
            fig_age = px.histogram(
                age_data, x="Age_years", nbins=12, title="Age at Diagnosis",
                color_discrete_sequence=["#3b82f6"], opacity=0.9
            )
            fig_age.update_layout(bargap=0.2)
            st.plotly_chart(fig_age, use_container_width=True)
            if selected_grade == "All":
                fig_age_grade = px.box(
                    filtered_df.dropna(subset=["Age_years"]),
                    x="Grade", y="Age_years", title="Age by Tumor Type",
                    color="Grade", color_discrete_sequence=["#10b981", "#ef4444"]
                )
                st.plotly_chart(fig_age_grade, use_container_width=True)
        else:
            st.warning("No ages available.")

    with st.expander("🧬 Top Genes"):
        st.markdown("**Discover the most mutated genes.**")
        mutation_freq = filtered_df[mutation_cols].apply(lambda x: (x == "MUTATED").sum())
        mutation_freq_df = pd.DataFrame({"Gene": mutation_freq.index, "Mutation Count": mutation_freq.values})
        fig = px.bar(
            mutation_freq_df.sort_values("Mutation Count", ascending=False).head(8),
            x="Gene", y="Mutation Count", title="Most Mutated Genes",
            color="Mutation Count", color_continuous_scale="Blues",
            hover_data={"Mutation Count": True}
        )
        fig.update_traces(marker_line_width=1.5, marker_line_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏥 Diagnoses"):
        st.markdown("**What types of gliomas are there?**")
        diagnosis_counts = filtered_df["Primary_Diagnosis"].value_counts().reset_index()
        diagnosis_counts.columns = ["Diagnosis", "Count"]
        fig_diag = px.pie(
            diagnosis_counts.head(5), names="Diagnosis", values="Count", 
            title="Common Diagnoses",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_diag.update_traces(textinfo="percent+label", pull=[0.1]*5)
        st.plotly_chart(fig_diag, use_container_width=True)

    with st.expander("👥 Gender"):
        st.markdown("**Compare mutations by gender.**")
        gender_df = filtered_df[filtered_df["Gender"].isin(["Female", "Male"])]
        if not gender_df.empty:
            gender_mutation = gender_df.groupby("Gender")[top_genes].apply(
                lambda x: (x == "MUTATED").sum()
            ).T.reset_index()
            gender_mutation.columns = ["Gene", "Female", "Male"]
            fig_gender = px.bar(
                gender_mutation.melt(id_vars="Gene", value_vars=["Female", "Male"], var_name="Gender", value_name="Count"),
                x="Gene", y="Count", color="Gender", barmode="group",
                title="Mutations by Gender",
                color_discrete_sequence=["#f472b6", "#60a5fa"]
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("No gender data for Female/Male.")

    with st.expander("🌍 Races"):
        st.markdown("**Patient backgrounds.**")
        race_counts = filtered_df["Race"].value_counts().reset_index()
        race_counts.columns = ["Race", "Count"]
        fig_race = px.bar(
            race_counts, x="Race", y="Count", title="Patients by Race",
            color="Count", color_continuous_scale="Purples"
        )
        st.plotly_chart(fig_race, use_container_width=True)

    with st.expander("⏳ Survival Hints"):
        st.markdown("**Age as a clue to outcomes.**")
        if selected_grade == "All":
            fig_survival = px.violin(
                filtered_df.dropna(subset=["Age_years"]),
                x="Grade", y="Age_years", title="Age by Tumor Type",
                color="Grade", box=True,
                color_discrete_sequence=["#10b981", "#ef4444"]
            )
            st.plotly_chart(fig_survival, use_container_width=True)
        else:
            st.info("Pick 'All' to compare tumor types.")

    with st.expander("💾 Save Data"):
        st.markdown("**Download your data.**")
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="tcga_glioma_filtered.csv",
            mime="text/csv",
            key="download_button"
        )
        st.markdown("**Preview**:")
        st.dataframe(filtered_df.head(10), use_container_width=True)

    # Footer
    st.markdown(
        """
        <hr style='border-color: #e2e8f0;'>
        <div style='text-align: center; color: #6b7280; font-family: "Inter", sans-serif;'>
            Built with Streamlit | TCGA Glioma Data | Made for Discovery
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.error(
        """
        😔 Missing `TCGA_GBM_LGG_Mutations_all.csv`. Please add it to the folder.
        """
    )
