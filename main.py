from google.cloud import storage
from actu import actu_str_gcs  

def calculate_insurance(request):
    """Google Cloud Function to calculate insurance cost."""
    try:
        # Parse the incoming JSON data
        data = request.get_json()

        # Extract inputs and other parameters
        inputs_data = data.get("inputs", {})
        age = data.get("age")
        policy_amount = data.get("policy_amount")
        payment_type = data.get("payment_type")

        # Validate inputs
        if not age or not policy_amount or not payment_type or not inputs_data:
            return {"error": "Missing 'age', 'policy_amount', 'payment_type', or 'inputs' in the request"}, 400

        # Prepare the inputs list
        inputs = [
            inputs_data['weight'], inputs_data['sex'], inputs_data['height'], inputs_data['sys_bp'],
            inputs_data['smoker'], inputs_data['nic_other'], inputs_data['num_meds'], inputs_data['occup_danger'],
            inputs_data['ls_danger'], inputs_data['cannabis'], inputs_data['opioids'], inputs_data['other_drugs'],
            inputs_data['drinks_aweek'], inputs_data['addiction'], inputs_data['major_surgery_num'],
            inputs_data['diabetes'], inputs_data['hds'], inputs_data['cholesterol'], inputs_data['asthma'],
            inputs_data['immune_defic'], inputs_data['family_cancer'], inputs_data['family_heart_disease'],
            inputs_data['family_cholesterol']
        ]

        # Call your existing actu_str_gcs function
        result = actu_str_gcs(inputs, policy_amount, age, payment_type)

        # Return the result as a JSON response
        return {"result": result}, 200

    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": str(e)}, 500
    
'''
curl -X POST https://us-central1-life-predictor-cpsc325.cloudfunctions.net/calculate_insurance \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "policy_amount": 150000,
    "payment_type": "Annual",
    "inputs": {
        "weight": 180,
        "sex": "m",
        "height": 72,
        "sys_bp": 120,
        "smoker": "n",
        "nic_other": "n",
        "num_meds": 2,
        "occup_danger": 1,
        "ls_danger": 2,
        "cannabis": "n",
        "opioids": "n",
        "other_drugs": "n",
        "drinks_aweek": 3,
        "addiction": "n",
        "major_surgery_num": 0,
        "diabetes": "n",
        "hds": "n",
        "cholesterol": 190,
        "asthma": "n",
        "immune_defic": "n",
        "family_cancer": "y",
        "family_heart_disease": "n",
        "family_cholesterol": "y"
    }
}'
'''