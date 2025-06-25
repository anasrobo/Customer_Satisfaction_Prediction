from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained ensemble model (ensure the path is correct)
ensemble_model = joblib.load('backend\models\ensemble_model.pkl')

# Define a mapping for the gender input
gender_mapping = {
    'Male': 0,
    'Female': 1,
    'Other': 2
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form values in the specified order
        customer_age = float(request.form['customer_age'])
        # Map the gender string to numeric
        customer_gender_str = request.form['customer_gender']
        customer_gender = gender_mapping.get(customer_gender_str)
        if customer_gender is None:
            raise ValueError("Invalid value for Customer Gender")
        
        product_purchased = float(request.form['product_purchased'])
        ticket_type = float(request.form['ticket_type'])
        ticket_subject = float(request.form['ticket_subject'])
        ticket_status = float(request.form['ticket_status'])
        resolution = float(request.form['resolution'])
        ticket_priority = float(request.form['ticket_priority'])
        ticket_channel = float(request.form['ticket_channel'])
        first_response_delay = float(request.form['first_response_delay'])
        resolution_time = float(request.form['resolution_time'])
        
        # Build a DataFrame with the exact feature names and order
        input_df = pd.DataFrame([{
            'Customer Age': customer_age,
            'Customer Gender': customer_gender,
            'Product Purchased': product_purchased,
            'Ticket Type': ticket_type,
            'Ticket Subject': ticket_subject,
            'Ticket Status': ticket_status,
            'Resolution': resolution,
            'Ticket Priority': ticket_priority,
            'Ticket Channel': ticket_channel,
            'First Response Delay (hrs)': first_response_delay,
            'Resolution Time (hrs)': resolution_time
        }])
        
        # Make prediction using the ensemble model
        prediction = ensemble_model.predict(input_df)
        prediction = int(prediction[0])
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)



    #       python flask_app\app\\app.py