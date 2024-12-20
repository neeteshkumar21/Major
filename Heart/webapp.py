import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/OneDrive/Desktop/Heart/SaveModel/heart_disease_model.sav', 'rb'))

def heart_disease(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'

def main():
    # Title for the web page
    st.title("Heart Disease Prediction Web App")

    # Getting input data from the user
    age = st.text_input("Age")
    sex = st.text_input("Sex (1 = Male, 0 = Female)")
    cp = st.text_input("Chest Pain Type (0-3)")
    trestbps = st.text_input("Resting Blood Pressure (mm Hg)")
    chol = st.text_input("Serum Cholesterol (mg/dL)")
    fbs = st.text_input("Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False)")
    restecg = st.text_input("Resting Electrocardiographic Results (0-2)")
    thalach = st.text_input("Maximum Heart Rate Achieved")
    exang = st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No)")
    oldpeak = st.text_input("ST Depression Induced by Exercise")
    slope = st.text_input("Slope of the Peak Exercise ST Segment (0-2)")
    ca = st.text_input("Number of Major Vessels Colored by Fluoroscopy (0-3)")
    thal = st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")

    # Code for prediction
    diagnosis = ''

    # Prediction and result display
    if st.button("Predict"):
        try:
            # Convert inputs to floats and make a list
            diagnosis = heart_disease([
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ])
            # Display result
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    main()
