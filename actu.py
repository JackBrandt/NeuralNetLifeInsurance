from neural_net import load_model
from utils import format_policy_type

def get_mortality_table(age, inputs):
    """Retrieve a mortality table from a neural network model based on age and inputs."""
    default_years = 25 - age if age < 25 else 0
    path='models/'+str(int(age+default_years))+'.pth'  # Correctly form the path string
    model = load_model(path)  # Ensure the path is passed here
    mortality_df = model.get_life_data([inputs], is_tensor=False, smooth=True, sigma=10)
    return mortality_df[0].to_numpy()

def calculate_present_value(fv, n, i):
    """Calculate the present value of a future payment given interest rate and number of years."""
    return fv * ((1 + i / 100) ** -n)

def calculate_annuity_payment(n, i, pv):
    """Calculate the payment for an annuity given duration, interest rate, and present value."""
    i = i / 100
    return pv / ((1 - (1 + i) ** -n) / i)

def calculate_annuity_present_value(n, i, pmt):
    """Calculate the present value of an annuity given number of periods, interest rate, and payment."""
    i = i / 100
    return pmt * ((1 - (1 + i) ** -n) / i)

def calculate_life_insurance_liability(fv, i, mortality_table, defer_years=0):
    """Calculate expected present value of life insurance liability."""
    pv = sum(calculate_present_value(fv, n, i) * mort for n, mort in enumerate(mortality_table, 1 + defer_years))
    return pv

def calculate_annual_to_monthly_payment(annual_payment, i):
    """Convert an annual payment to a monthly payment considering compounding interest."""
    monthly_i = ((1 + i / 100) ** (1 / 12) - 1) * 100
    return annual_payment / calculate_annuity_present_value(12, monthly_i, 1)

def actuarial_string(inputs, fv, age, policy_type='fl', duration=None, payment_type=None, i=1):
    """Generate a descriptive string about an insurance policy."""
    default_years = 25 - age if age < 25 else 0
    mortality_table = get_mortality_table(age, inputs)
    liability_pv = calculate_life_insurance_liability(fv, i, mortality_table, default_years)
    
    if payment_type == 'Lump':
        return f"A ${fv:,.2f} life {format_policy_type(policy_type, duration).lower()} policy for this {age} year-old person would cost a ${liability_pv:.2f} lump payment up front."
    elif payment_type == 'Annual':
        annuity_pv = calculate_annuity_present_value(len(mortality_table), i, 1)
        pmt = liability_pv / annuity_pv
        return f"A ${fv:,.2f} life {format_policy_type(policy_type, duration).lower()} policy for this {age} year-old person would cost an annual payment of ${pmt:.2f}."
    elif payment_type == 'Monthly':
        annual_pmt = calculate_annuity_present_value(len(mortality_table), i, 1)
        monthly_pmt = calculate_annual_to_monthly_payment(annual_pmt, i)
        return f"A ${fv:,.2f} life {format_policy_type(policy_type, duration).lower()} policy for this {age} year-old person would cost a monthly payment of ${monthly_pmt:.2f}."
    else:
        return "Unknown payment type provided."

if __name__ == "__main__":
    # Example usage
    print(actuarial_string([180, 'm', 72, 130, 'n', 'n', 3, 1, 1, 'n', 'n', 'n', 4, 'n', 0, 'n', 'n', 200, 'n', 'n', 'n', 'n', 'n'], 250000, 20, payment_type='Annual'))
