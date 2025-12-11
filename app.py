"""
CFST Compressive Strength Prediction and Inverse Design Web Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CFST Prediction",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }

    .sub-header {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Card style containers */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
    }

    /* Prediction result box */
    .prediction-result {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 5px solid #0ea5e9;
        margin: 1.5rem 0;
        text-align: center;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #0369a1;
        margin: 0.5rem 0;
    }

    .prediction-unit {
        font-size: 1.2rem;
        color: #64748b;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.875rem 1.5rem;
        border-radius: 0.75rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 8px -1px rgba(239, 68, 68, 0.4);
        transform: translateY(-1px);
    }

    /* Sidebar styling */
    .sidebar-title {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        color: white;
    }

    .sidebar-title h3 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.4;
    }

    /* Parameter selection cards */
    .param-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid transparent;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }

    .param-card:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }

    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #3b82f6;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .section-header h3 {
        margin: 0;
        color: #1e3a5f;
        font-size: 1.25rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }

    .footer p {
        margin: 0.25rem 0;
        font-size: 0.875rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Training message */
    .training-msg {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }

    /* Input section */
    .input-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk, or train if not available"""
    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    model_path = Path(f'models/catboost_{model_name}.pkl')
    stats_path = Path(f'models/stats_{model_name}.pkl')

    if model_path.exists() and stats_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        return model, stats

    # Train model on first run
    data = load_data()
    if data is None:
        return None, None

    model_configs = {
        'N': {
            'target': 'P_exp (kN)',
            'features': ['D (mm)', 't  (mm)', 'f_y (MPa)', 'f_c (MPa)', 'L (mm)', 'e_t (mm)'],
            'params': {'iterations': 784, 'depth': 5, 'learning_rate': 0.2048,
                      'random_strength': 1.519e-07, 'bagging_temperature': 0.9238,
                      'loss_function': 'Tweedie:variance_power=1.75', 'l2_leaf_reg': 62,
                      'random_state': 0, 'verbose': False}
        },
        'D': {
            'target': 'D (mm)',
            'features': ['t  (mm)', 'f_y (MPa)', 'f_c (MPa)', 'L (mm)', 'e_t (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        },
        't': {
            'target': 't  (mm)',
            'features': ['D (mm)', 'f_y (MPa)', 'f_c (MPa)', 'L (mm)', 'e_t (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        },
        'fy': {
            'target': 'f_y (MPa)',
            'features': ['D (mm)', 't  (mm)', 'f_c (MPa)', 'L (mm)', 'e_t (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        },
        'fc': {
            'target': 'f_c (MPa)',
            'features': ['D (mm)', 't  (mm)', 'f_y (MPa)', 'L (mm)', 'e_t (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        },
        'L': {
            'target': 'L (mm)',
            'features': ['D (mm)', 't  (mm)', 'f_y (MPa)', 'f_c (MPa)', 'e_t (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        },
        'et': {
            'target': 'e_t (mm)',
            'features': ['D (mm)', 't  (mm)', 'f_y (MPa)', 'f_c (MPa)', 'L (mm)', 'P_exp (kN)'],
            'params': {'iterations': 1000, 'depth': 4, 'learning_rate': 0.1241,
                      'random_strength': 1.1948973774883714e-06, 'bagging_temperature': 1,
                      'loss_function': 'Tweedie:variance_power=1.9', 'l2_leaf_reg': 2,
                      'random_state': 42, 'verbose': False}
        }
    }

    config = model_configs.get(model_name)
    if config is None:
        st.error(f"Unknown model: {model_name}")
        return None, None

    X = data[config['features']]
    y = data[config['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = CatBoostRegressor(**config['params'])
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    stats = {
        'feature_names': config['features'],
        'target_name': config['target'],
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, model.predict(X_train)),
        'train_rmse': np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
        'feature_ranges': {
            feature: {
                'min': float(data[feature].min()),
                'max': float(data[feature].max()),
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std())
            } for feature in config['features']
        }
    }

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    return model, stats

@st.cache_data
def load_data():
    """Load the CFST dataset"""
    data_path = Path('data/1287dataCCFT.csv')
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace('$', '').str.replace('_{', '_').str.replace('}', '').str.strip()
    return df

# Sidebar
st.sidebar.markdown("""
<div class='sidebar-title'>
    <h3>CFST Column Strength Prediction and Inverse Design</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Data Visualization", "Original Data"],
    format_func=lambda x: {"Prediction": "🎯 Prediction", "Data Visualization": "📊 Visualization", "Original Data": "📋 Dataset"}[x],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# About section in sidebar
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    This application predicts CFST column properties using CatBoost machine learning models trained on 1,287 experimental samples.

    **Features:**
    - Forward prediction (strength)
    - Inverse design (dimensions)
    - Data visualization
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**Contact:**")
st.sidebar.markdown("📧 [khuongln@utt.edu.vn](mailto:khuongln@utt.edu.vn)")
st.sidebar.markdown("📧 [saeed.banihashemi@canberra.edu.au](mailto:saeed.banihashemi@canberra.edu.au)")

# Main content
if page == "Prediction":
    st.markdown("<h1 class='main-header'>CFST Column Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Circular Concrete-Filled Steel Tube Compressive Strength & Inverse Design</p>", unsafe_allow_html=True)

    # Two-column layout for parameter selection
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Select Output Parameter")

        parameter_options = {
            "N": ("Compressive Strength N", "kN", "🔴"),
            "D": ("Diameter D", "mm", "🔵"),
            "t": ("Steel Thickness t", "mm", "🟢"),
            "fy": ("Steel Yield Strength fy", "MPa", "🟡"),
            "fc": ("Concrete Strength fc", "MPa", "🟠"),
            "L": ("Column Height L", "mm", "🟣"),
            "et": ("Load Eccentricity et", "mm", "⚪")
        }

        selected_code = st.radio(
            "Parameter to predict:",
            list(parameter_options.keys()),
            format_func=lambda x: f"{parameter_options[x][2]} {parameter_options[x][0]} ({parameter_options[x][1]})",
            label_visibility="collapsed"
        )

        selected_name, selected_unit, _ = parameter_options[selected_code]

        # Show what we're predicting
        st.info(f"**Predicting:** {selected_name}")

    with col_right:
        # Load model
        with st.spinner(f"Loading {selected_name} model..."):
            model, stats = load_model(selected_code)

        if model is not None and stats is not None:
            st.markdown("#### Input Parameters")

            input_values = {}

            param_configs = {
                't  (mm)': {'label': 'Steel Tube Thickness t', 'unit': 'mm', 'min': 1.0, 'max': 17.0, 'default': 4.0, 'step': 0.1},
                'D (mm)': {'label': 'Diameter D', 'unit': 'mm', 'min': 50.0, 'max': 550.0, 'default': 114.0, 'step': 1.0},
                'L (mm)': {'label': 'Height L', 'unit': 'mm', 'min': 152.0, 'max': 5400.0, 'default': 1369.0, 'step': 10.0},
                'f_y (MPa)': {'label': 'Steel Yield Strength fy', 'unit': 'MPa', 'min': 186.0, 'max': 1153.0, 'default': 352.0, 'step': 1.0},
                'e_t (mm)': {'label': 'Load Eccentricity et', 'unit': 'mm', 'min': 0.0, 'max': 341.0, 'default': 13.0, 'step': 1.0},
                'f_c (MPa)': {'label': 'Concrete Strength fc', 'unit': 'MPa', 'min': 9.0, 'max': 186.0, 'default': 51.0, 'step': 1.0},
                'P_exp (kN)': {'label': 'Target Strength N', 'unit': 'kN', 'min': 14.0, 'max': 46000.0, 'default': 1982.0, 'step': 10.0}
            }

            features = stats['feature_names']

            # Create 2 columns for sliders
            slider_col1, slider_col2 = st.columns(2)

            for i, feature in enumerate(features):
                config = param_configs.get(feature, {
                    'label': feature, 'unit': '',
                    'min': float(stats['feature_ranges'][feature]['min']),
                    'max': float(stats['feature_ranges'][feature]['max']),
                    'default': float(stats['feature_ranges'][feature]['mean']),
                    'step': 0.1
                })

                container = slider_col1 if i % 2 == 0 else slider_col2
                with container:
                    value = st.slider(
                        f"{config['label']} ({config['unit']})",
                        min_value=config['min'],
                        max_value=config['max'],
                        value=config['default'],
                        step=config['step'],
                        key=f"slider_{feature}"
                    )
                    input_values[feature] = value

            st.markdown("")

            # Prediction button
            if st.button("🎯 Run Prediction", use_container_width=True):
                input_data = pd.DataFrame([input_values])
                prediction = model.predict(input_data)[0]

                # Display result
                st.markdown(f"""
                <div class='prediction-result'>
                    <p style='margin: 0; color: #64748b; font-size: 1rem;'>Predicted {selected_name}</p>
                    <p class='prediction-value'>{prediction:,.2f}</p>
                    <p class='prediction-unit'>{selected_unit}</p>
                </div>
                """, unsafe_allow_html=True)

                # Model performance metrics
                st.markdown("##### Model Performance")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Train R²", f"{stats['train_r2']:.4f}")
                m2.metric("Test R²", f"{stats['test_r2']:.4f}")
                m3.metric("Test RMSE", f"{stats['test_rmse']:.2f}")
                m4.metric("Samples", "1,287")

                # Input summary
                with st.expander("📋 Input Summary"):
                    input_df = pd.DataFrame({
                        'Parameter': [param_configs.get(f, {'label': f})['label'] for f in input_values.keys()],
                        'Value': list(input_values.values()),
                        'Unit': [param_configs.get(f, {'unit': ''})['unit'] for f in input_values.keys()]
                    })
                    st.dataframe(input_df, use_container_width=True, hide_index=True)

elif page == "Data Visualization":
    st.markdown("<h1 class='main-header'>Data Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore the CFST experimental dataset</p>", unsafe_allow_html=True)

    data = load_data()

    if data is not None:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", f"{len(data):,}")
        c2.metric("Features", len(data.columns) - 1)
        c3.metric("Min Strength", f"{data['P_exp (kN)'].min():,.0f} kN")
        c4.metric("Max Strength", f"{data['P_exp (kN)'].max():,.0f} kN")

        st.markdown("---")

        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔥 Correlations", "📈 Statistics"])

        with tab1:
            feature = st.selectbox("Select feature:", data.columns, key="dist_feature")

            col1, col2 = st.columns(2)

            with col1:
                fig_hist = px.histogram(
                    data, x=feature, nbins=30,
                    title=f"Distribution of {feature}",
                    color_discrete_sequence=['#3b82f6']
                )
                fig_hist.update_layout(
                    showlegend=False,
                    xaxis_title=feature,
                    yaxis_title="Frequency",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                fig_box = px.box(
                    data, y=feature,
                    title=f"Box Plot of {feature}",
                    color_discrete_sequence=['#ef4444']
                )
                fig_box.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)

        with tab2:
            corr_matrix = data.corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 11},
                colorbar=dict(title="Correlation")
            ))

            fig_corr.update_layout(
                title="Feature Correlation Matrix",
                height=550,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Scatter plot
            st.markdown("##### Scatter Plot")
            sc1, sc2 = st.columns(2)
            x_feat = sc1.selectbox("X-axis:", data.columns, index=0)
            y_feat = sc2.selectbox("Y-axis:", data.columns, index=len(data.columns)-1)

            fig_scatter = px.scatter(
                data, x=x_feat, y=y_feat,
                color='P_exp (kN)',
                color_continuous_scale='Viridis',
                title=f"{y_feat} vs {x_feat}"
            )
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.markdown("##### Statistical Summary")
            st.dataframe(data.describe().round(2), use_container_width=True)

elif page == "Original Data":
    st.markdown("<h1 class='main-header'>Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Browse and filter the experimental dataset</p>", unsafe_allow_html=True)

    data = load_data()

    if data is not None:
        # Filter controls
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            filter_feature = st.selectbox("Filter by:", ['None'] + list(data.columns))

        filtered_data = data.copy()

        if filter_feature != 'None':
            with col2:
                min_val, max_val = float(data[filter_feature].min()), float(data[filter_feature].max())
                filter_range = st.slider(
                    f"Range for {filter_feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered_data = data[
                    (data[filter_feature] >= filter_range[0]) &
                    (data[filter_feature] <= filter_range[1])
                ]

        with col3:
            st.metric("Showing", f"{len(filtered_data):,} / {len(data):,}")

        st.markdown("---")

        # Data table
        st.dataframe(
            filtered_data,
            use_container_width=True,
            height=450
        )

        # Download
        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
        with col_dl2:
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="cfst_data.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("""
<div class='footer'>
    <p><strong>CFST Column Prediction Tool</strong></p>
    <p>Powered by CatBoost ML | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
