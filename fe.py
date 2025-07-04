import streamlit as st
import requests


# Set page configuration
st.set_page_config(
    page_title="Churn Prediction MLOps",
    page_icon="🔄",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8800"  

def main():
    st.title("Churn Prediction MLOps Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Churn", "Retrain Model"])
    
    if page == "Predict Churn":
        predict_churn_page()
    elif page == "Retrain Model":
        retrain_model_page()

def predict_churn_page():
    st.header("Predict Customer Churn")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
    
    # Model version selection
    col1, col2 = st.columns(2)
    with col1:
        model_version = st.text_input("Model Version", value="1")
    with col2:
        scaler_version = st.text_input("Scaler Version", value="scaler/scaler_churn_version_20250705T001416.pkl")
    
    # Run ID input
    col1, col2 = st.columns(2)
    with col1:
        run_id = st.text_input("Run ID", value="1fef47d0e3fc4b40b3732437f41716ae")
    with col2:
        reference_data = st.text_input(
            "Reference Data (Optional)", 
            value="s3://churndataversion/churn_data_store/data_version/features_data_version_20250704T155040.csv"
        )
    
    # Submit button
    if st.button("Run Prediction"):
        if uploaded_file is not None:
            with st.spinner("Processing, get update in https://dagshub.com/Teungtran/churn_mlops.mlflow ..."):
                try:
                    # Prepare the form data
                    files = {"file": uploaded_file}
                    form_data = {
                        "model_version": model_version,
                        "scaler_version": scaler_version,
                        "run_id": run_id,
                        "reference_data": reference_data
                    }
                    
                    # Make the API call
                    response = requests.post(
                        f"{API_URL}/churn/",
                        files=files,
                        data=form_data
                    )
                    
                    if response.status_code == 200:
                        st.success("Prediction request submitted successfully!")
                        
                        # Get the payload from response
                        response_data = response.json()
                        payload = response_data.get("payload", {})
                        
                        # Display the results
                        st.subheader("Prediction Results")
                        
                        # Display message
                        if payload.get("message"):
                            st.markdown("**Analysis Summary:**")
                            st.info(payload["message"])
                        
                        if payload.get("prediction_url"):
                            st.markdown("**Prediction File:**")
                            st.success(f"Results are available at, click to download: {payload['prediction_url']}")
                            
                        
                        if payload.get("timestamp"):
                            st.markdown("**Processing Time:**")
                            st.text(f"Completed at: {payload['timestamp']}")
                        
                        st.markdown("---")
                        st.info("""
                        **Next Steps:**
                        - Results are saved to MLflow experiment 'Churn_model_prediction_cycle'
                        - Prediction file is uploaded to S3 bucket
                        - Check the MLflow dashboard for detailed metrics and visualizations
                        """)
                        
                            
                    else:
                        st.error(f"Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Please make sure the server is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning("Please upload a file first")

def retrain_model_page():
    st.header("Retrain Churn Model")
    
    # Check workflow status
    if st.button("Check Workflow Status"):
        with st.spinner("Checking workflow status..."):
            try:
                response = requests.get(f"{API_URL}/workflow/status")
                if response.status_code == 200:
                    status_data = response.json()
                    st.json(status_data)
                    
                    # Display a more user-friendly message
                    if status_data.get("status") == "ready":
                        if status_data.get("data_file_exists"):
                            st.success("Workflow system is ready with existing data")
                        else:
                            st.warning("Workflow system is ready but requires data upload")
                else:
                    st.error(f"Error checking status: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    st.subheader("Retrain with New Data")
    
    uploaded_file = st.file_uploader("Upload new training data (Optional)", type=["csv", "xlsx"])
    
    # Submit button for retraining
    if st.button("Start Retraining"):
        with st.spinner("Retraining model... This may take several minutes"):
            try:
                files = {"file": uploaded_file} if uploaded_file else None
                
                response = requests.post(
                    f"{API_URL}/workflow/train",
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Model retraining completed successfully!")
                    st.json(result)
                    
                    if result.get("final_model_path"):
                        st.info(f"Final model saved to: {result['final_model_path']}")
                else:
                    st.error(f"Error during retraining: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()