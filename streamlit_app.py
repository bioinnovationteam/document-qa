"""
Biodesign Needs Screener
A tool for biodesign students to assess market context for medical needs
"""

import sys
import subprocess
import importlib.util

# Check and install required packages
required_packages = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'requests': 'requests',
    'plotly': 'plotly',
    'sklearn': 'scikit-learn'
}

missing_packages = []
for package_name, pip_name in required_packages.items():
    if importlib.util.find_spec(package_name) is None:
        missing_packages.append(pip_name)

if missing_packages:
    print("=" * 60)
    print("Missing required packages. Installing now...")
    print("=" * 60)
    for package in missing_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("\nAll packages installed! Please restart the app.")
    print("Run: streamlit run biodesign_screener.py")
    sys.exit(0)

# Import
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


# Page config
st.set_page_config(
    page_title="Biodesign Needs Screener",
    page_icon="F",
    layout="wide"
)

# Title and description
st.title("Biodesign Needs Screening Tool")
st.markdown("""
This tool helps biodesign students quickly assess implementation context for their needs statements.
    NOTE: This tools is not omniscient, do not rely on this for a thorough analysis report.
Enter a medical need below to see relevant CPT codes, FDA product codes, and market context.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    st.markdown("**Data Sources (all public)**")
    st.markdown("- FDA Product Classification Database")
    st.markdown("- CMS Procedure Price Lookup")
    st.markdown("- USPTO Open Data Portal")
    st.markdown("- NIH Clinical Tables")
    
    st.divider()
    
    st.markdown("**About**")
    st.markdown("Built for biodesign students to accelerate needs screening, invention, and implementation.")
    st.markdown("All data is from public sources. Not for clinical use.")
    st.markdown("Copyright 2026 BioInnovation Team LLC")


# Main input area
col1, col2 = st.columns([3, 1])
with col1:
    needs_statement = st.text_area(
        "Enter your needs statement:",
        "EXAMPLE: A better way to close the left atrial appendage during cardiac surgery to reduce stroke risk in AFib patients",
        height=100
    )
with col2:
    run_analysis = st.button("Analyze Need", type="primary", use_container_width=True)

# Mock data functions (replace with real APIs later)
@st.cache_data
def load_cpt_codes():
    """Load CPT codes from local file or create mock"""
    # In production, load from CMS API or downloaded CSV
    cpt_data = pd.DataFrame({
        'cpt_code': ['33340', '33267', '33268', '33269', '33270'],
        'description': [
            'Percutaneous left atrial appendage closure',
            'Ligation of left atrial appendage',
            'Excision of left atrial appendage',
            'Transcatheter left atrial appendage closure',
            'Insertion of left atrial appendage device'
        ],
        'category': ['Surgery', 'Surgery', 'Surgery', 'Surgery', 'Surgery'],
        'work_rvu': [12.5, 10.2, 11.8, 13.1, 14.2]
    })
    return cpt_data

@st.cache_data
def load_fda_product_codes():
    """Load FDA product codes"""
    fda_data = pd.DataFrame({
        'product_code': ['NMD', 'DQY', 'OKD', 'MJP', 'LNK'],
        'device_name': [
            'Left Atrial Appendage Closure Device',
            'Cardiac Occlusion Device',
            'Atrial Septal Defect Occluder',
            'Patent Foramen Ovale Occluder',
            'Vascular Occlusion Device'
        ],
        'regulation_number': ['870.3925', '870.3800', '870.3800', '870.3800', '870.3800'],
        'device_class': ['III', 'II', 'II', 'II', 'II'],
        'panel': ['Cardiovascular', 'Cardiovascular', 'Cardiovascular', 'Cardiovascular', 'Cardiovascular']
    })
    return fda_data

@st.cache_data
def load_510k_predicates():
    """Load mock 510(k) predicate devices"""
    predicates = pd.DataFrame({
        'k_number': ['K210162', 'K200657', 'K191482', 'K183125', 'K172145'],
        'device_name': [
            'Watchman FLX Left Atrial Appendage Closure Device',
            'Amplatzer Amulet Left Atrial Appendage Occluder',
            'Lambre Left Atrial Appendage Occluder',
            'WaveCrest Left Atrial Appendage Occlusion System',
            'Watchman Left Atrial Appendage Closure Device'
        ],
        'applicant': [
            'Boston Scientific',
            'Abbott Medical',
            'Lifetech Scientific',
            'Johnson & Johnson',
            'Boston Scientific'
        ],
        'clearance_date': [
            '2021-07-20',
            '2020-03-15',
            '2019-08-22',
            '2018-11-30',
            '2017-04-12'
        ],
        'product_code': ['NMD', 'NMD', 'NMD', 'NMD', 'NMD']
    })
    return predicates

@st.cache_data
def load_patent_data():
    """Load mock patent data"""
    patents = pd.DataFrame({
        'patent_number': ['US10835228B2', 'US10660647B2', 'US10398450B2', 'US10292711B2', 'US10076340B2'],
        'title': [
            'Left atrial appendage closure device',
            'Left atrial appendage occluder',
            'Left atrial appendage closure systems',
            'Left atrial appendage closure method',
            'Left atrial appendage occlusion device'
        ],
        'assignee': [
            'Boston Scientific',
            'Abbott Medical',
            'Lifetech Scientific',
            'Johnson & Johnson',
            'Boston Scientific'
        ],
        'filing_year': [2018, 2017, 2016, 2015, 2014],
        'status': ['Active', 'Active', 'Active', 'Expired', 'Active']
    })
    return patents

@st.cache_data
def load_cms_pricing():
    """Load CMS pricing data"""
    pricing = pd.DataFrame({
        'cpt_code': ['33340', '33267', '33268', '33269', '33270'],
        'hospital_outpatient': [18500, 12200, 13500, 19800, 21000],
        'ambulatory_surgical_center': [15200, 9800, 10800, 16200, 17500],
        'physician_payment': [1250, 950, 1050, 1350, 1450],
        'volume_2023': [45000, 8200, 3500, 12700, 22300],
        'growth_rate': [0.12, -0.03, -0.08, 0.15, 0.21]
    })
    return pricing

# Semantic expansion function (mock)
def expand_needs_statement(text):
    """Use Natural Language Processing to expand the needs statement with related terms"""
    # In production, use actual NLP model
    # This is a simple keyword extractor for demo
    import re
    from collections import Counter
    
    # Simple tokenization and keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Medical term expansions (mock)
    expansions = {
        'close': ['closure', 'occlusion', 'seal', 'ligation', 'exclusion'],
        'atrial': ['atrial', 'atrium', 'auricular'],
        'appendage': ['appendage', 'auricle', 'ligament'],
        'cardiac': ['cardiac', 'heart', 'cardiovascular'],
        'surgery': ['surgery', 'surgical', 'procedure', 'intervention'],
        'stroke': ['stroke', 'cva', 'embolism', 'thrombosis'],
        'afib': ['atrial fibrillation', 'afib', 'arrhythmia']
    }
    
    expanded = set()
    for word in words:
        if word in expansions:
            expanded.update(expansions[word])
    
    return list(expanded)

# Main analysis function
def analyze_need(needs_statement):
    """Run the full analysis pipeline"""
    
    # Step 1: Semantic expansion
    expanded_terms = expand_needs_statement(needs_statement)
    
    # Step 2: Load reference data
    cpt_df = load_cpt_codes()
    fda_df = load_fda_product_codes()
    predicates_df = load_510k_predicates()
    patents_df = load_patent_data()
    pricing_df = load_cms_pricing()
    
    # Step 3: Find relevant CPT codes (cosine similarity)
    # In production, use actual vector embeddings
    # For demo, just return all and let user decide
    relevant_cpt = cpt_df
    
    # Step 4: Find relevant FDA product codes
    # Simple keyword matching for demo
    relevant_fda = fda_df  # In production, use semantic search
    
    # Step 5: Get predicates for those product codes
    relevant_predicates = predicates_df[predicates_df['product_code'].isin(relevant_fda['product_code'])]
    
    # Step 6: Get patents (mock - just return all)
    relevant_patents = patents_df
    
    # Step 7: Get pricing for relevant CPT codes
    relevant_pricing = pricing_df[pricing_df['cpt_code'].isin(relevant_cpt['cpt_code'])]
    
    # Step 8: Calculate simple opportunity scores
    scores = []
    for _, row in relevant_pricing.iterrows():
        cpt = row['cpt_code']
        volume = row['volume_2023']
        payment = row['hospital_outpatient']
        
        # Simple scoring: market size (volume * payment) with penalty for competition
        market_size = volume * payment / 1000000  # in millions
        competition = len(relevant_predicates)  # crude proxy
        patent_density = len(relevant_patents)  # crude proxy
        
        opportunity = (market_size * 0.5) / (competition * 0.3 + patent_density * 0.2 + 1)
        
        scores.append({
            'cpt_code': cpt,
            'market_size_millions': round(market_size, 1),
            'competition_count': competition,
            'patent_count': patent_density,
            'opportunity_score': round(opportunity, 1)
        })
    
    scores_df = pd.DataFrame(scores)
    
    return {
        'expanded_terms': expanded_terms,
        'cpt_codes': relevant_cpt,
        'fda_codes': relevant_fda,
        'predicates': relevant_predicates,
        'patents': relevant_patents,
        'pricing': relevant_pricing,
        'scores': scores_df
    }

# Run analysis when button clicked
if run_analysis and needs_statement:
    with st.spinner("Analyzing needs statement..."):
        # Add small delay to simulate processing
        time.sleep(1.5)
        results = analyze_need(needs_statement)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary", 
        "CPT Codes", 
        "FDA Context", 
        "Market Data", 
        "Opportunity Score"
    ])
    
    with tab1:
        st.header("Needs Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Need")
            st.info(needs_statement)
            
            st.subheader("Expanded Search Terms")
            st.write(", ".join(results['expanded_terms']))
        
        with col2:
            st.subheader("Quick Stats")
            stats_df = pd.DataFrame({
                'Metric': ['Relevant CPT Codes', 'FDA Product Codes', 'Predicate Devices', 'Active Patents'],
                'Count': [
                    len(results['cpt_codes']),
                    len(results['fda_codes']),
                    len(results['predicates']),
                    len(results['patents'])
                ]
            })
            st.dataframe(stats_df, hide_index=True)
        
        st.divider()
        
        st.subheader("Top 3 Predicate Devices to Investigate")
        top_predicates = results['predicates'].head(3)
        for _, pred in top_predicates.iterrows():
            with st.expander(f"{pred['device_name']} ({pred['k_number']})"):
                st.write(f"**Applicant:** {pred['applicant']}")
                st.write(f"**Cleared:** {pred['clearance_date']}")
                st.write(f"**Product Code:** {pred['product_code']}")
                st.markdown(f"[Search FDA Database](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={pred['k_number']})")
    
    with tab2:
        st.header("Relevant CPT Codes")
        st.markdown("These procedure codes map to your needs statement. Use them to research reimbursement and utilization.")
        
        # Merge with pricing data
        cpt_with_pricing = results['cpt_codes'].merge(
            results['pricing'][['cpt_code', 'hospital_outpatient', 'volume_2023']], 
            on='cpt_code', 
            how='left'
        )
        
        st.dataframe(
            cpt_with_pricing,
            column_config={
                'cpt_code': 'CPT Code',
                'description': 'Description',
                'work_rvu': 'Work RVU',
                'hospital_outpatient': st.column_config.NumberColumn('Hospital Payment ($)', format="$%d"),
                'volume_2023': 'Annual Volume'
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.caption("Data source: CMS Physician Fee Schedule (mock data for demo)")
    
    with tab3:
        st.header("FDA Regulatory Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Classifications")
            st.dataframe(
                results['fda_codes'],
                column_config={
                    'product_code': 'Product Code',
                    'device_name': 'Device Type',
                    'device_class': 'Class',
                    'panel': 'Panel'
                },
                hide_index=True
            )
        
        with col2:
            st.subheader("Predicate Devices (510(k) Clearances)")
            st.dataframe(
                results['predicates'],
                column_config={
                    'k_number': 'K Number',
                    'device_name': 'Device',
                    'applicant': 'Company',
                    'clearance_date': 'Cleared',
                    'product_code': 'Product Code'
                },
                hide_index=True,
                height=300
            )
        
        st.subheader("Patent Landscape")
        # Simple chart of patents by year
        patent_years = results['patents'].groupby('filing_year').size().reset_index(name='count')
        fig = px.bar(patent_years, x='filing_year', y='count', 
                     title='Patent Filings by Year',
                     labels={'filing_year': 'Year', 'count': 'Number of Patents'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Patents")
        st.dataframe(
            results['patents'].head(5),
            column_config={
                'patent_number': 'Patent',
                'title': 'Title',
                'assignee': 'Assignee',
                'filing_year': 'Year'
            },
            hide_index=True
        )
    
    with tab4:
        st.header("Market Context")
        
        # Merge pricing with CPT descriptions
        market_data = results['pricing'].merge(
            results['cpt_codes'][['cpt_code', 'description']],
            on='cpt_code'
        )
        
        st.subheader("Procedure Volume & Payment")
        
        # Bubble chart: volume vs payment
        fig = px.scatter(
            market_data,
            x='volume_2023',
            y='hospital_outpatient',
            size='volume_2023',
            text='description',
            title='Procedure Market Size: Volume vs Payment',
            labels={'volume_2023': 'Annual Volume', 'hospital_outpatient': 'Hospital Payment ($)'}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Pricing Data")
        st.dataframe(
            market_data,
            column_config={
                'cpt_code': 'CPT',
                'description': 'Procedure',
                'hospital_outpatient': st.column_config.NumberColumn('Hospital ($)', format="$%d"),
                'ambulatory_surgical_center': st.column_config.NumberColumn('ASC ($)', format="$%d"),
                'physician_payment': st.column_config.NumberColumn('Physician ($)', format="$%d"),
                'volume_2023': '2023 Volume',
                'growth_rate': st.column_config.NumberColumn('Growth', format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )
    
    with tab5:
        st.header("Opportunity Scoring")
        st.markdown("""
        This is a simple composite score to compare opportunities across CPT codes.
        **Higher score = potentially more attractive opportunity.**
        """)
        
        # Sort by opportunity score
        sorted_scores = results['scores'].sort_values('opportunity_score', ascending=False)
        
        # Bar chart of opportunity scores
        fig = px.bar(
            sorted_scores,
            x='cpt_code',
            y='opportunity_score',
            title='Relative Opportunity by CPT Code',
            labels={'cpt_code': 'CPT Code', 'opportunity_score': 'Opportunity Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Score Breakdown")
        st.dataframe(
            sorted_scores,
            column_config={
                'cpt_code': 'CPT Code',
                'market_size_millions': 'Market Size ($M)',
                'competition_count': 'Competitors',
                'patent_count': 'Patents',
                'opportunity_score': 'Score'
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.caption("""
        **How scoring works:** Market size (volume × payment) is positive; 
        competition and patent density are negative factors. 
        Weighted: Market size (0.5), Competition (0.3), Patents (0.2).
        """)
    
    # Footer with export option
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        csv = results['cpt_codes'].to_csv(index=False)
        st.download_button(
            label="📥 Export Results (CSV)",
            data=csv,
            file_name=f"biodesign_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # Show example when no analysis run
    st.info("Enter a needs statement and click 'Analyze Need' to begin")
    
    st.markdown("""
    ### What this tool does:
    1. **Semantically expands** your needs statement into search terms
    2. **Maps to CPT codes** for procedure reimbursement research
    3. **Finds FDA product codes** and predicate devices
    4. **Shows patent activity** in the space
    5. **Calculates opportunity scores** to compare across codes
    
    Built with Streamlit, Plotly, and public data sources.
    """)